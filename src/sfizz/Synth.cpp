// SPDX-License-Identifier: BSD-2-Clause

// This code is part of the sfizz library and is licensed under a BSD 2-clause
// license. You should have receive a LICENSE.md file along with the code.
// If not, contact the sfizz maintainers at https://github.com/sfztools/sfizz

#include "SynthPrivate.h"
#include "Config.h"
#include "utility/Debug.h"
#include "utility/Macros.h"
#include "utility/U8Strings.h"
#include "modulations/ModId.h"
#include "modulations/ModKey.h"
#include "modulations/ModMatrix.h"
#include "PolyphonyGroup.h"
#include "pugixml.hpp"
#include "Region.h"
#include "RegionSet.h"
#include "ReleaseAlign.h"
#include "DivisionGate.h"
#include "Composite.h"
#include "Resources.h"
#include "BufferPool.h"
#include "FilePool.h"
#include "Wavetables.h"
#include "Tuning.h"
#include "BeatClock.h"
#include "Metronome.h"
#include "SynthConfig.h"
#include "ScopedFTZ.h"
#include "utility/Base64.h"
#include "utility/StringViewHelpers.h"
#include "utility/Timing.h"
#include "utility/XmlHelpers.h"
#include "Voice.h"
#include "ADSREnvelope.h"
#include "Interpolators.h"
#include "parser/Parser.h"
#include <absl/algorithm/container.h>
#include <absl/memory/memory.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_replace.h>
#include <absl/types/optional.h>
#include <absl/types/span.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <utility>

namespace sfz {

// unless set to permissive, the loader rejects sfz files with errors
static constexpr bool loaderParsesPermissively = true;

Synth::Synth()
: impl_(new Impl) // NOLINT: (paul) I don't get why clang-tidy complains here
{
}

// Need to define the dtor after Impl has been defined
Synth::~Synth()
{

}

Synth::Impl::Impl()
{
    initializeSIMDDispatchers();
    initializeInterpolators();

    parser_.setListener(this);
    effectFactory_.registerStandardEffectTypes();
    initEffectBuses();
    resetVoices(config::numVoices);
    resetDefaultCCValues();
    resetAllControllers(0);

    // modulation sources
    MidiState& midiState = resources_.getMidiState();
    genController_.reset(new ControllerSource(resources_, voiceManager_));
    genLFO_.reset(new LFOSource(voiceManager_));
    genFlexEnvelope_.reset(new FlexEnvelopeSource(voiceManager_));
    genADSREnvelope_.reset(new ADSREnvelopeSource(voiceManager_));
    genChannelAftertouch_.reset(new ChannelAftertouchSource(voiceManager_, midiState));
    genPolyAftertouch_.reset(new PolyAftertouchSource(voiceManager_, midiState));
}

Synth::Impl::~Impl()
{
    voiceManager_.reset();
    resources_.getFilePool().emptyFileLoadingQueues();
}

void Synth::Impl::onParseFullBlock(const std::string& header, const std::vector<Opcode>& members)
{
    const auto newRegionSet = [&](OpcodeScope level) {
        auto parent = currentSet_;
        while (parent && parent->getLevel() >= level)
            parent = parent->getParent();

        sets_.emplace_back(new RegionSet(parent, level));
        currentSet_ = sets_.back().get();
    };

    switch (hash(header)) {
    case hash("global"):
        globalOpcodes_ = members;
        newRegionSet(OpcodeScope::kOpcodeScopeGlobal);
        groupOpcodes_.clear();
        masterOpcodes_.clear();
        handleGlobalOpcodes(members);
        break;
    case hash("control"):
        defaultPath_ = ""; // Always reset on a new control header
        handleControlOpcodes(members);
        break;
    case hash("master"):
        masterOpcodes_ = members;
        newRegionSet(OpcodeScope::kOpcodeScopeMaster);
        groupOpcodes_.clear();
        handleMasterOpcodes(members);
        numMasters_++;
        break;
    case hash("group"):
        groupOpcodes_ = members;
        newRegionSet(OpcodeScope::kOpcodeScopeGroup);
        handleGroupOpcodes(members, masterOpcodes_);
        numGroups_++;
        break;
    case hash("region"):
        buildRegion(members);
        break;
    case hash("curve"):
        resources_.getCurves().addCurveFromHeader(members);
        break;
    case hash("effect"):
        handleEffectOpcodes(members);
        break;
    case hash("sample"):
        handleSampleOpcodes(members);
        break;
    default:
        std::cerr << "Unknown header: " << header << '\n';
    }
}

void Synth::Impl::onParseError(const SourceRange& range, const std::string& message)
{
    const auto relativePath = range.start.filePath->lexically_relative(parser_.originalDirectory());
    std::cerr << "Parse error in " << relativePath << " at line " << range.start.lineNumber + 1 << ": " << message << '\n';
}

void Synth::Impl::onParseWarning(const SourceRange& range, const std::string& message)
{
    const auto relativePath = range.start.filePath->lexically_relative(parser_.originalDirectory());
    std::cerr << "Parse warning in " << relativePath << " at line " << range.start.lineNumber + 1 << ": " << message << '\n';
}

void Synth::Impl::buildRegion(const std::vector<Opcode>& regionOpcodes)
{
    int regionNumber = static_cast<int>(layers_.size());
    MidiState& midiState = resources_.getMidiState();
    Layer* lastLayer = new Layer(regionNumber, defaultPath_, midiState);
    layers_.emplace_back(lastLayer);
    Region* lastRegion = &lastLayer->getRegion();

    //
    auto parseOpcodes = [&](const std::vector<Opcode>& opcodes) {
        for (auto& opcode : opcodes) {
            const auto unknown = absl::c_find_if(unknownOpcodes_, [&](absl::string_view sv) { return sv.compare(opcode.name) == 0; });
            if (unknown != unknownOpcodes_.end()) {
                continue;
            }

            if (!lastRegion->parseOpcode(opcode))
                unknownOpcodes_.emplace_back(opcode.name);
        }
    };

    parseOpcodes(globalOpcodes_);
    parseOpcodes(masterOpcodes_);
    parseOpcodes(groupOpcodes_);
    parseOpcodes(regionOpcodes);

    // Create the amplitude envelope
    if (!lastRegion->flexAmpEG)
        lastRegion->getOrCreateConnection(
            ModKey::createNXYZ(ModId::AmpEG, lastRegion->id),
            ModKey::createNXYZ(ModId::MasterAmplitude, lastRegion->id)).sourceDepth = 1.0f;
    else
        lastRegion->getOrCreateConnection(
            ModKey::createNXYZ(ModId::Envelope, lastRegion->id, *lastRegion->flexAmpEG),
            ModKey::createNXYZ(ModId::MasterAmplitude, lastRegion->id)).sourceDepth = 1.0f;

    if (octaveOffset_ != 0 || noteOffset_ != 0)
        lastRegion->offsetAllKeys(octaveOffset_ * 12 + noteOffset_);

    if (lastRegion->lastKeyswitch)
        lastKeyswitchLists_[*lastRegion->lastKeyswitch].push_back(lastLayer);

    if (lastRegion->lastKeyswitchRange) {
        auto& range = *lastRegion->lastKeyswitchRange;
        for (uint8_t note = range.getStart(), end = range.getEnd(); note <= end; note++)
            lastKeyswitchLists_[note].push_back(lastLayer);
    }

    if (lastRegion->upKeyswitch)
        upKeyswitchLists_[*lastRegion->upKeyswitch].push_back(lastLayer);

    if (lastRegion->downKeyswitch)
        downKeyswitchLists_[*lastRegion->downKeyswitch].push_back(lastLayer);

    if (lastRegion->previousKeyswitch)
        previousKeyswitchLists_.push_back(lastLayer);

    if (lastRegion->defaultSwitch)
        setCurrentSwitch(*lastRegion->defaultSwitch);

    // There was a combination of group= and polyphony= on a region, so set the group polyphony
    if (lastRegion->group != Default::group && lastRegion->polyphony != config::maxVoices) {
        voiceManager_.setGroupPolyphony(lastRegion->group, lastRegion->polyphony);
    } else {
        // Just check that there are enough polyphony groups
        voiceManager_.ensureNumPolyphonyGroups(lastRegion->group);
    }

    if (currentSet_ != nullptr) {
        lastRegion->parent = currentSet_;
        currentSet_->addRegion(lastRegion);
    }

    // Adapt the size of the delayed releases to avoid allocating later on
    if (lastRegion->trigger == Trigger::release) {
        const auto keyLength = static_cast<unsigned>(lastRegion->keyRange.length());
        const auto size = max(config::delayedReleaseVoices, keyLength);
        lastLayer->delayedSustainReleases_.reserve(size);
        lastLayer->delayedSostenutoReleases_.reserve(size);
    }

    // Initialize status of Key switches, CC switches, etc
    lastLayer->initializeActivations();
}

void Synth::Impl::addEffectBusesIfNecessary(uint16_t output)
{
    while (effectBuses_.size() <= output) {
        // Add output
        effectBuses_.emplace_back();
        auto& buses = effectBuses_.back();
        // Add an empty main bus on output
        buses.emplace_back(new EffectBus);
        buses[0]->setGainToMain(1.0);
        buses[0]->setSamplesPerBlock(samplesPerBlock_);
        buses[0]->setSampleRate(sampleRate_);
        buses[0]->clearInputs(samplesPerBlock_);
    }
}

void Synth::Impl::initEffectBuses()
{
    effectBuses_.clear();
    addEffectBusesIfNecessary(0);
}

void Synth::Impl::clear()
{
    FilePool& filePool = resources_.getFilePool();
    MidiState& midiState = resources_.getMidiState();

    // Clear the background queues before removing everyone
    filePool.waitForBackgroundLoading();

    voiceManager_.reset();
    for (auto& list : lastKeyswitchLists_)
        list.clear();
    for (auto& list : downKeyswitchLists_)
        list.clear();
    for (auto& list : upKeyswitchLists_)
        list.clear();
    for (auto& list : noteActivationLists_)
        list.clear();
    for (auto& list : compositeLists_)
        list.clear();
    for (auto& list : compositeOctaveLists_)
        list.clear();
    compositeOctavesById_.clear();
    for (auto& list : ccActivationLists_)
        list.clear();
    previousKeyswitchLists_.clear();

    currentSet_ = nullptr;
    sets_.clear();
    layers_.clear();
    resources_.clearNonState();
    rootPath_.clear();
    numGroups_ = 0;
    numMasters_ = 0;
    numOutputs_ = 1;
    noteOffset_ = 0;
    octaveOffset_ = 0;
    currentSwitch_ = absl::nullopt;
    defaultPath_ = "";
    image_ = "";
    midiState.resetNoteStates();
    midiState.flushEvents();
    filePool.setRamLoading(config::loadInRam);
    clearCCLabels();
    currentUsedCCs_.clear();
    sustainOrSostenuto_.clear();
    changedCCsThisCycle_.clear();
    changedCCsLastCycle_.clear();
    clearKeyLabels();
    keySlots_.clear();
    swLastSlots_.clear();
    clearKeyswitchLabels();
    globalOpcodes_.clear();
    masterOpcodes_.clear();
    groupOpcodes_.clear();
    unknownOpcodes_.clear();
    modificationTime_ = absl::nullopt;
    playheadMoved_ = false;

    initEffectBuses();
}

void Synth::Impl::handleMasterOpcodes(const std::vector<Opcode>& members)
{
    for (auto& rawMember : members) {
        const Opcode member = rawMember.cleanUp(kOpcodeScopeMaster);

        switch (member.lettersOnlyHash) {
        case hash("polyphony"):
            ASSERT(currentSet_ != nullptr);
            currentSet_->setPolyphonyLimit(member.read(Default::polyphony));
            break;
        case hash("sw_default"):
            setCurrentSwitch(member.read(Default::key));
            break;
        }
    }
}

void Synth::Impl::handleGlobalOpcodes(const std::vector<Opcode>& members)
{
    for (auto& rawMember : members) {
        const Opcode member = rawMember.cleanUp(kOpcodeScopeGlobal);

        switch (member.lettersOnlyHash) {
        case hash("polyphony"):
            ASSERT(currentSet_ != nullptr);
            currentSet_->setPolyphonyLimit(member.read(Default::polyphony));
            break;
        case hash("sw_default"):
            setCurrentSwitch(member.read(Default::key));
            break;
        case hash("volume"):
            // FIXME : Probably best not to mess with this and let the host control the volume
            // setValueFromOpcode(member, volume, OldDefault::volumeRange);
            break;
        }
    }
}

void Synth::Impl::handleGroupOpcodes(const std::vector<Opcode>& members, const std::vector<Opcode>& masterMembers)
{
    absl::optional<int64_t> groupIdx;
    absl::optional<unsigned> maxPolyphony;

    const auto parseOpcode = [&](const Opcode& rawMember) {
        const Opcode member = rawMember.cleanUp(kOpcodeScopeGroup);

        switch (member.lettersOnlyHash) {
        case hash("group"):
            groupIdx = member.read(Default::group);
            break;
        case hash("polyphony"):
            maxPolyphony = member.read(Default::polyphony);
            break;
        case hash("sw_default"):
            setCurrentSwitch(member.read(Default::key));
            break;
        }
    };

    for (auto& member : masterMembers)
        parseOpcode(member);

    for (auto& member : members)
        parseOpcode(member);

    if (groupIdx && maxPolyphony) {
        voiceManager_.setGroupPolyphony(*groupIdx, *maxPolyphony);
    } else if (maxPolyphony) {
        ASSERT(currentSet_ != nullptr);
        currentSet_->setPolyphonyLimit(*maxPolyphony);
    } else if (groupIdx) {
        voiceManager_.ensureNumPolyphonyGroups(*groupIdx);
    }
}

void Synth::Impl::handleControlOpcodes(const std::vector<Opcode>& members)
{
    for (auto& rawMember : members) {
        const Opcode member = rawMember.cleanUp(kOpcodeScopeControl);

        switch (member.lettersOnlyHash) {
        case hash("set_cc&"):
            if (Default::ccNumber.bounds.containsWithEnd(member.parameters.back())) {
                const auto ccNumber = member.parameters.back();
                const auto value = member.read(Default::loCC);
                setDefaultHdcc(ccNumber, value);
                if (!reloading)
                    resources_.getMidiState().ccEvent(0, ccNumber, value);
            }
            break;
        case hash("set_hdcc&"):
            if (Default::ccNumber.bounds.containsWithEnd(member.parameters.back())) {
                const auto ccNumber = member.parameters.back();
                const auto value = member.read(Default::loNormalized);
                setDefaultHdcc(ccNumber, value);
                if (!reloading)
                    resources_.getMidiState().ccEvent(0, ccNumber, value);
            }
            break;
        case hash("label_cc&"):
            if (Default::ccNumber.bounds.containsWithEnd(member.parameters.back()))
                setCCLabel(member.parameters.back(), std::string(member.value));
            break;
        case hash("label_key&"):
            if (member.parameters.back() <= Default::key.bounds.getEnd()) {
                const auto noteNumber = static_cast<uint8_t>(member.parameters.back());
                setKeyLabel(noteNumber, member.value);
            }
            break;
        case hash("default_path"):
            defaultPath_ = absl::StrReplaceAll(trim(member.value), { { "\\", "/" } });
            DBG("Changing default sample path to " << defaultPath_);
            break;
        case hash("image"):
            image_ = absl::StrCat(defaultPath_, absl::StrReplaceAll(trim(member.value), { { "\\", "/" } }));
            break;
        case hash("image_controls"):
            image_controls_ = absl::StrCat(defaultPath_, absl::StrReplaceAll(trim(member.value), { { "\\", "/" } }));
            break;
        case hash("note_offset"):
            noteOffset_ = member.read(Default::noteOffset);
            break;
        case hash("octave_offset"):
            octaveOffset_ = member.read(Default::octaveOffset);
            break;
        case hash("hint_ram_based"):
        {
            FilePool& filePool = resources_.getFilePool();
            filePool.setRamLoading(member.read(Default::ramBased));
        } break;
        case hash("hint_stealing"):
            switch(hash(member.value)) {
            case hash("first"):
                voiceManager_.setStealingAlgorithm(StealingAlgorithm::First);
                break;
            case hash("oldest"):
                voiceManager_.setStealingAlgorithm(StealingAlgorithm::Oldest);
                break;
            case hash("envelope_and_age"):
                voiceManager_.setStealingAlgorithm(StealingAlgorithm::EnvelopeAndAge);
                break;
            default:
                DBG("Unsupported value for hint_stealing: " << member.value);
            }
            break;
        case hash("hint_sustain_cancels_release"):
        {
            SynthConfig& config = resources_.getSynthConfig();
            config.sustainCancelsRelease = member.read(Default::sustainCancelsRelease);
        }
            break;
        default:
            // Unsupported control opcode
            DBG("Unsupported control opcode: " << member.name);
        }
    }
}


void Synth::Impl::handleEffectOpcodes(const std::vector<Opcode>& rawMembers)
{
    absl::string_view busName { "main" };
    uint16_t output { Default::output };

    std::vector<Opcode> members;
    members.reserve(rawMembers.size());
    for (const Opcode& opcode : rawMembers) {
        if (opcode.lettersOnlyHash == hash("output"))
            output = opcode.read(Default::output);

        members.push_back(opcode.cleanUp(kOpcodeScopeEffect));
    }

    addEffectBusesIfNecessary(output);

    auto getOrCreateBus = [this, output](unsigned index) -> EffectBus& {
        if (index + 1 > effectBuses_[output].size())
            effectBuses_[output].resize(index + 1);
        EffectBusPtr& bus = effectBuses_[output][index];
        if (!bus) {
            bus.reset(new EffectBus);
            bus->setSampleRate(sampleRate_);
            bus->setSamplesPerBlock(samplesPerBlock_);
            bus->clearInputs(samplesPerBlock_);
        }
        return *bus;
    };

    for (const Opcode& opcode : members) {
        switch (opcode.lettersOnlyHash) {
        case hash("bus"):
            busName = opcode.value;
            break;

            // note(jpc): gain opcodes are linear volumes in % units

        case hash("directtomain"):
            getOrCreateBus(0).setGainToMain(opcode.read(Default::effect));
            break;

        case hash("fx&tomain"): // fx&tomain
            {
                const auto busIndex = opcode.parameters.front();
                if (busIndex < 1 || busIndex > config::maxEffectBuses)
                    break;
                getOrCreateBus(busIndex).setGainToMain(opcode.read(Default::effect));
            }
            break;

        case hash("fx&tomix"): // fx&tomix
            {
                const auto busIndex = opcode.parameters.front();
                if (busIndex < 1 || busIndex > config::maxEffectBuses)
                    break;
                getOrCreateBus(busIndex).setGainToMix(opcode.read(Default::effect));
            }
            break;
        }
    }

    unsigned busIndex;
    if (busName.empty() || busName == "main")
        busIndex = 0;
    else if (busName.size() > 2 && busName.substr(0, 2) == "fx" && absl::SimpleAtoi(busName.substr(2), &busIndex) && busIndex >= 1 && busIndex <= config::maxEffectBuses) {
        // an effect bus fxN, with N usually in [1,4]
    } else {
        DBG("Unsupported effect bus: " << busName);
        return;
    }

    // create the effect and add it
    auto fx = effectFactory_.makeEffect(members);
    fx->setSampleRate(sampleRate_);
    fx->setSamplesPerBlock(samplesPerBlock_);
    getOrCreateBus(busIndex).addEffect(std::move(fx));
}

void Synth::Impl::handleSampleOpcodes(const std::vector<Opcode>& rawMembers)
{
    absl::string_view name { "" };
    bool hasData { false };
    absl::string_view sampleData;

    for (const Opcode& opcode : rawMembers) {
        switch (opcode.lettersOnlyHash) {
        case hash("name"):
            name = opcode.value;
            break;
        case hash("base&data"):
            if (opcode.parameters.front() == 64)
                sampleData = opcode.value;
            break;
        case hash("data"):
            hasData = true;
            break;
        }
    }

    if (name.empty())
        return;

    if (hasData && sampleData.empty()) {
        DBG("The sample data provided for sample " << name
            << " doesn't use base64 encoding, which is the only one sfizz knows how to decode.\n "
            << "If it does, please use base64data= instead of data=."
        );
        return;
    }

    if (sampleData.empty())
        return;

    auto data = decodeBase64(sampleData);
    FilePool& filePool = resources_.getFilePool();
    FileId id { std::string(name) };
    filePool.loadFromRam(id, data);
}

void Synth::Impl::resetDefaultCCValues() noexcept
{
    fill(absl::MakeSpan(defaultCCValues_), 0.0f);
    setDefaultHdcc(7, normalizeCC(100));
    setDefaultHdcc(10, 0.5f);
    setDefaultHdcc(11, 1.0f);

    setCCLabel(7, "Volume");
    setCCLabel(10, "Pan");
    setCCLabel(11, "Expression");
}

void Synth::Impl::prepareSfzLoad(const fs::path& path)
{
    auto newPath_ = path.string();
    reloading = (lastPath_ == newPath_);

    clear();

#ifndef NDEBUG
    if (reloading) {
        DBG("[sfizz] Reloading the current file");
    }
#endif

    if (!reloading) {

        // Clear the background queues and clear the filePool
        auto& filePool = resources_.getFilePool();
        filePool.waitForBackgroundLoading();
        filePool.clear();

        // Set the default hdcc to their default
        resetDefaultCCValues();

        // Store the new path
        lastPath_ = std::move(newPath_);
    }
}

bool Synth::loadSfzFile(const fs::path& file)
{
    Impl& impl = *impl_;
    impl.prepareSfzLoad(file);

    std::error_code ec;
    fs::path realFile = fs::canonical(file, ec);
    bool success = true;
    Parser& parser = impl.parser_;
    parser.parseFile(ec ? file : realFile);

    // permissive parsing for compatibility
    if (!loaderParsesPermissively)
        success = parser.getErrorCount() == 0;

    success = success && !impl.layers_.empty();

    if (!success) {
        DBG("[sfizz] Loading failed");
        auto& filePool = impl.resources_.getFilePool();
        parser.clear();
        filePool.clear();
        return false;
    }

    impl.finalizeSfzLoad();
    return true;
}

bool Synth::loadSfzString(const fs::path& path, absl::string_view text)
{
    Impl& impl = *impl_;
    impl.prepareSfzLoad(path);

    bool success = true;
    Parser& parser = impl.parser_;
    parser.parseString(path, text);

    // permissive parsing for compatibility
    if (!loaderParsesPermissively)
        success = parser.getErrorCount() == 0;

    success = success && !impl.layers_.empty();

    if (!success) {
        auto& filePool = impl.resources_.getFilePool();
        DBG("[sfizz] Loading failed");
        parser.clear();
        filePool.clear();
        return false;
    }

    impl.finalizeSfzLoad();
    return true;
}

void Synth::Impl::setCurrentSwitch(uint8_t noteValue)
{
    currentSwitch_ = noteValue + 12 * octaveOffset_ + noteOffset_;
}

void Synth::Impl::finalizeSfzLoad()
{
    FilePool& filePool = resources_.getFilePool();
    WavetablePool& wavePool = resources_.getWavePool();

    const fs::path& rootDirectory = parser_.originalDirectory();
    filePool.setRootDirectory(rootDirectory);

    // a string representation used for OSC purposes
    rootPath_ = u8EncodedString(rootDirectory);

    size_t currentRegionIndex = 0;
    size_t currentRegionCount = layers_.size();

    // per file: the largest offset, and the head its regions ask for (0 = the pool's size)
    absl::flat_hash_map<sfz::FileId, std::pair<int64_t, int64_t>> filesToLoad;

    auto removeCurrentRegion = [this, &currentRegionIndex, &currentRegionCount]() {
        const Region& region = layers_[currentRegionIndex]->getRegion();
        DBG("Removing the region with sample " << *region.sampleId);
        layers_.erase(layers_.begin() + currentRegionIndex);
        --currentRegionCount;
    };

    size_t maxFilters { 0 };
    size_t maxEQs { 0 };
    size_t maxLFOs { 0 };
    size_t maxFlexEGs { 0 };
    bool havePitchEG { false };
    bool haveFilterEG { false };
    bool haveAmplitudeLFO { false };
    bool havePitchLFO { false };
    bool haveFilterLFO { false };

    FlexEGs::clearUnusedCurves();

    while (currentRegionIndex < currentRegionCount) {
        Layer& layer = *layers_[currentRegionIndex];
        Region& region = layer.getRegion();

        absl::optional<FileInformation> fileInformation;

        if (!region.isGenerator()) {
            if (!filePool.checkSampleId(*region.sampleId)) {
                removeCurrentRegion();
                continue;
            }

            fileInformation = filePool.getFileInformation(*region.sampleId);
            if (!fileInformation) {
                removeCurrentRegion();
                continue;
            }

            region.hasWavetableSample = fileInformation->wavetable.has_value();

            if (fileInformation->end < config::wavetableMaxFrames) {
                auto sample = filePool.loadFile(*region.sampleId);
                bool allZeros = true;
                int numChannels = sample->information.numChannels;
                for (int i = 0; i < numChannels; ++i) {
                    allZeros &= allWithin(sample->preloadedData.getConstSpan(i),
                        -config::virtuallyZero, config::virtuallyZero);
                }

                if (allZeros) {
                    region.sampleId.reset(new FileId("*silence"));
                    region.hasWavetableSample = false;
                }
            }
        }

        if (!region.isOscillator()) {
            region.sampleEnd = min(region.sampleEnd, fileInformation->end);

            if (fileInformation->hasLoop) {
                if (region.loopRange.getStart() == Default::loopStart)
                    region.loopRange.setStart(fileInformation->loopStart);

                if (region.loopRange.getEnd() == Default::loopEnd)
                    region.loopRange.setEnd(fileInformation->loopEnd);

                if (!region.loopMode)
                    region.loopMode = LoopMode::loop_continuous;
            }

            if (region.isRelease() && !region.loopMode)
                region.loopMode = LoopMode::one_shot;

            if (region.loopRange.getEnd() == Default::loopEnd)
                region.loopRange.setEnd(region.sampleEnd);

            // if range is invalid, disable the loop
            if (!region.loopRange.isValid())
                region.loopMode = absl::nullopt;

            if (fileInformation->numChannels == 2)
                region.hasStereoSample = true;

            if (region.pitchKeycenterFromSample)
                region.pitchKeycenter = fileInformation->rootKey;

            // TODO: adjust with LFO targets
            const auto maxOffset = [region]() {
                uint64_t sumOffsetCC = region.offset + region.offsetRandom;
                for (const auto& offsets : region.offsetCC)
                    sumOffsetCC += offsets.data;
                return Default::offsetMod.bounds.clamp(sumOffsetCC);
            }();

            auto& toLoad = filesToLoad[*region.sampleId];
            toLoad.first = max(toLoad.first, maxOffset);
            toLoad.second = max(toLoad.second, region.headFrames);
        }
        else if (!region.isGenerator()) {
            if (!wavePool.createFileWave(filePool, std::string(region.sampleId->filename()))) {
                removeCurrentRegion();
                continue;
            }
        }

        if (region.lastKeyswitch) {
            if (currentSwitch_)
                layer.keySwitched_ = (*currentSwitch_ == *region.lastKeyswitch);

            if (region.keyswitchLabel)
                setKeyswitchLabel(*region.lastKeyswitch, *region.keyswitchLabel);
        }

        if (region.lastKeyswitchRange) {
            auto& range = *region.lastKeyswitchRange;
            if (currentSwitch_)
                layer.keySwitched_ = range.containsWithEnd(*currentSwitch_);

            if (region.keyswitchLabel) {
                for (uint8_t note = range.getStart(), end = range.getEnd(); note <= end; note++)
                    setKeyswitchLabel(note, *region.keyswitchLabel);
            }
        }

        for (auto note = 0; note < 128; note++) {
            if (!region.keyRange.containsWithEnd(note))
                continue;
            // Composite regions (Composite.h) are started only by the composite path:
            // the attacks are listed apart, the summed tails nowhere.
            if (region.compositeId < 0)
                noteActivationLists_[note].push_back(&layer);
            else if (region.isRelease())
                continue;
            else if (region.compositeOctaves != 0)
                compositeOctaveLists_[note].push_back(&layer);
            else
                compositeLists_[note].push_back(&layer);
        }

        for (int cc = 0; cc < config::numCCs; cc++) {
            if (region.ccTriggers.contains(cc)
                || region.ccConditions.contains(cc)
                || (cc == region.sustainCC && region.trigger == Trigger::release)
                || (cc == region.sostenutoCC && region.trigger == Trigger::release))
                ccActivationLists_[cc].push_back(&layer);
        }

        // Defaults
        MidiState& midiState = resources_.getMidiState();
        for (int cc = 0; cc < config::numCCs; cc++) {
            layer.updateCCState(cc, midiState.getCCValue(cc));
        }


        // Set the default frequencies on equalizers if needed
        if (region.equalizers.size() > 0
            && region.equalizers[0].frequency == Default::eqFrequency) {
            region.equalizers[0].frequency = Default::defaultEQFreq[0];
            if (region.equalizers.size() > 1
                && region.equalizers[1].frequency == Default::eqFrequency) {
                region.equalizers[1].frequency = Default::defaultEQFreq[1];
                if (region.equalizers.size() > 2
                    && region.equalizers[2].frequency == Default::eqFrequency) {
                    region.equalizers[2].frequency = Default::defaultEQFreq[2];
                }
            }
        }

        if (!region.velocityPoints.empty())
            region.velCurve = Curve::buildFromVelcurvePoints(
                region.velocityPoints, Curve::Interpolator::Linear);

        layer.registerPitchWheel(midiState.getPitchBend());
        layer.registerAftertouch(midiState.getChannelAftertouch());
        layer.registerTempo(static_cast<float>(resources_.getBeatClock().getBeatsPerSecond()));
        layer.registerProgramChange(midiState.getProgram());
        maxFilters = max(maxFilters, region.filters.size());
        maxEQs = max(maxEQs, region.equalizers.size());
        maxLFOs = max(maxLFOs, region.lfos.size());
        maxFlexEGs = max(maxFlexEGs, region.flexEGs.size());
        havePitchEG = havePitchEG || region.pitchEG != absl::nullopt;
        haveFilterEG = haveFilterEG || region.filterEG != absl::nullopt;
        haveAmplitudeLFO = haveAmplitudeLFO || region.amplitudeLFO != absl::nullopt;
        havePitchLFO = havePitchLFO || region.pitchLFO != absl::nullopt;
        haveFilterLFO = haveFilterLFO || region.filterLFO != absl::nullopt;
        numOutputs_ = max(region.output + 1, numOutputs_);

        ++currentRegionIndex;
    }

    // Reset the preload call count to check for unused preloaded samples
    // when reloading
    if (reloading)
        filePool.resetPreloadCallCounts();

    {
        // Batch + parallel-decode the preloads (organdroid patch): same semantics as
        // per-file preloadFile(), but a cold load uses every core and reports progress.
        std::vector<FilePool::PreloadRequest> preloadList;
        preloadList.reserve(filesToLoad.size());
        for (const auto& toLoad: filesToLoad)
            preloadList.push_back({ toLoad.first, static_cast<uint32_t>(toLoad.second.first),
                                    static_cast<uint32_t>(toLoad.second.second) });
        filePool.preloadFiles(preloadList);
    }

    // Remove preloaded data with no linked regions
    if (reloading)
        filePool.removeUnusedPreloadedData();

    // Remove bad regions with unknown files
    if (currentRegionCount < layers_.size()) {
        DBG("Removing " << (layers_.size() - currentRegionCount)
            << " out of " << layers_.size() << " regions");
    }
    layers_.resize(currentRegionCount);
    resolveComposites();

    // collect all CCs used in regions, with matrix not yet connected
    BitArray<config::numCCs> usedCCs;
    for (const LayerPtr& layerPtr : layers_) {
        const Region& region = layerPtr->getRegion();
        collectUsedCCsFromRegion(usedCCs, region);
        for (const Region::Connection& connection : region.connections) {
            if (connection.source.id() == ModId::Controller)
                usedCCs.set(connection.source.parameters().cc);
        }
    }
    // connect default controllers, except if these CC are already used
    for (const LayerPtr& layerPtr : layers_) {
        Region& region = layerPtr->getRegion();
        constexpr unsigned defaultSmoothness = 10;
        if (!usedCCs.test(7)) {
            region.getOrCreateConnection(
                ModKey::createCC(7, 4, defaultSmoothness, 0),
                ModKey::createNXYZ(ModId::Amplitude, region.id)).sourceDepth = 1.0f;
        }
        if (!usedCCs.test(10)) {
            region.getOrCreateConnection(
                ModKey::createCC(10, 1, defaultSmoothness, 0),
                ModKey::createNXYZ(ModId::Pan, region.id)).sourceDepth = 1.0f;
        }
        if (!usedCCs.test(11)) {
            region.getOrCreateConnection(
                ModKey::createCC(11, 4, defaultSmoothness, 0),
                ModKey::createNXYZ(ModId::Amplitude, region.id)).sourceDepth = 1.0f;
        }
    }

    modificationTime_ = checkModificationTime();

    settingsPerVoice_.maxFilters = maxFilters;
    settingsPerVoice_.maxEQs = maxEQs;
    settingsPerVoice_.maxLFOs = maxLFOs;
    settingsPerVoice_.maxFlexEGs = maxFlexEGs;
    settingsPerVoice_.havePitchEG = havePitchEG;
    settingsPerVoice_.haveFilterEG = haveFilterEG;
    settingsPerVoice_.haveAmplitudeLFO = haveAmplitudeLFO;
    settingsPerVoice_.havePitchLFO = havePitchLFO;
    settingsPerVoice_.haveFilterLFO = haveFilterLFO;

    applySettingsPerVoice();
    addEffectBusesIfNecessary(numOutputs_);
    setupModMatrix();

    // cache the set of used CCs for future access
    currentUsedCCs_ = collectAllUsedCCs();

    // cache the set of keys assigned
    for (const LayerPtr& layerPtr : layers_) {
        const Region& region = layerPtr->getRegion();
        UncheckedRange<uint8_t> keyRange = region.keyRange;
        unsigned loKey = keyRange.getStart();
        unsigned hiKey = keyRange.getEnd();
        for (unsigned key = loKey; key <= hiKey; ++key)
            keySlots_.set(key);
    }
    // cache the set of keyswitches assigned
    for (const LayerPtr& layerPtr : layers_) {
        const Region& region = layerPtr->getRegion();
        if (absl::optional<uint8_t> sw = region.lastKeyswitch) {
            swLastSlots_.set(*sw);
        }
        else if (absl::optional<UncheckedRange<uint8_t>> swRange = region.lastKeyswitchRange) {
            unsigned loKey = swRange->getStart();
            unsigned hiKey = swRange->getEnd();
            for (unsigned key = loKey; key <= hiKey; ++key)
                swLastSlots_.set(key);
        }
    }
}

bool Synth::loadScalaFile(const fs::path& path)
{
    Impl& impl = *impl_;
    return impl.resources_.getTuning().loadScalaFile(path);
}

bool Synth::loadScalaString(const std::string& text)
{
    Impl& impl = *impl_;
    return impl.resources_.getTuning().loadScalaString(text);
}

void Synth::setScalaRootKey(int rootKey)
{
    Impl& impl = *impl_;
    impl.resources_.getTuning().setScalaRootKey(rootKey);
}

int Synth::getScalaRootKey() const
{
    Impl& impl = *impl_;
    return impl.resources_.getTuning().getScalaRootKey();
}

void Synth::setTuningFrequency(float frequency)
{
    Impl& impl = *impl_;
    impl.resources_.getTuning().setTuningFrequency(frequency);
}

float Synth::getTuningFrequency() const
{
    Impl& impl = *impl_;
    return impl.resources_.getTuning().getTuningFrequency();
}

void Synth::loadStretchTuningByRatio(float ratio)
{
    Impl& impl = *impl_;
    SFIZZ_CHECK(ratio >= 0.0f && ratio <= 1.0f);
    ratio = clamp(ratio, 0.0f, 1.0f);

    absl::optional<StretchTuning>& stretch = impl.resources_.getStretch();
    if (ratio > 0.0f)
        stretch = StretchTuning::createRailsbackFromRatio(ratio);
    else
        stretch.reset();
}

int Synth::getNumActiveVoices() const noexcept
{
    Impl& impl = *impl_;

    int activeVoices = static_cast<int>(impl.voiceManager_.getNumActiveVoices());

    // do not count overflow voices which are over limit
    return (config::overflowVoiceMultiplier > 1) ?
        std::min(impl.numVoices_, activeVoices) : activeVoices;
}

std::vector<const Voice*> Synth::getActiveVoices() const noexcept
{
    Impl& impl = *impl_;
    return impl.voiceManager_.getActiveVoices();
}

void Synth::setSamplesPerBlock(int samplesPerBlock) noexcept
{
    Impl& impl = *impl_;
    ASSERT(samplesPerBlock <= config::maxBlockSize);

    impl.samplesPerBlock_ = samplesPerBlock;
    for (auto& voice : impl.voiceManager_)
        voice.setSamplesPerBlock(samplesPerBlock);

    impl.resources_.setSamplesPerBlock(samplesPerBlock);

    for (int i = 0; i < impl.numOutputs_; ++i) {
        for (auto& bus : impl.getEffectBusesForOutput(i)) {
            if (bus)
                bus->setSamplesPerBlock(samplesPerBlock);
        }
    }
}

int Synth::getSamplesPerBlock() const noexcept
{
    Impl& impl = *impl_;
    return impl.samplesPerBlock_;
}

void Synth::setSampleRate(float sampleRate) noexcept
{
    Impl& impl = *impl_;

    impl.sampleRate_ = sampleRate;
    for (auto& voice : impl.voiceManager_)
        voice.setSampleRate(sampleRate);

    impl.resources_.setSampleRate(sampleRate);

    for (int i = 0; i < impl.numOutputs_; ++i) {
        for (auto& bus : impl.getEffectBusesForOutput(i)) {
            if (bus)
                bus->setSampleRate(sampleRate);
        }
    }
}

void Synth::renderBlock(AudioSpan<float> buffer) noexcept
{
    Impl& impl = *impl_;
    ScopedFTZ ftz;
    auto& callbackBreakdown = impl.callbackBreakdown_;
    impl.resetCallbackBreakdown();
    callbackBreakdown.dispatch = impl.dispatchDuration_;
    impl.dispatchDuration_ = 0.0;

    { // Silence buffer
        ScopedTiming logger { callbackBreakdown.renderMethod };
        buffer.fill(0.0f);
    }

    const size_t numFrames = buffer.getNumFrames();
    if (numFrames < 1) {
        CHECKFALSE;
        return;
    }

    const SynthConfig& synthConfig = impl.resources_.getSynthConfig();
    FilePool& filePool = impl.resources_.getFilePool();
    BufferPool& bufferPool = impl.resources_.getBufferPool();

    if (synthConfig.freeWheeling)
        filePool.waitForBackgroundLoading();

    impl.framesSinceGarbageCollection_ += buffer.getNumFrames();
    if (FilePool::garbageCollectionDue(impl.framesSinceGarbageCollection_, impl.sampleRate_)) {
        impl.framesSinceGarbageCollection_ = 0;
        filePool.triggerGarbageCollection();
    }

    auto tempSpan = bufferPool.getStereoBuffer(numFrames);
    auto tempMixSpan = bufferPool.getStereoBuffer(numFrames);
    auto rampSpan = bufferPool.getBuffer(numFrames);
    if (!tempSpan || !tempMixSpan || !rampSpan) {
        DBG("[sfizz] Could not get a temporary buffer; exiting callback... ");
        return;
    }

    // Composite stops (Composite.h): a composite voice that would play past its handover
    // time in this block hands over to its member pipes before the block, so the members
    // take over at a block boundary, as live note events start.
    impl.voiceManager_.forEachActive([&](Voice& voice) {
        const Layer* layer = voice.getCompositeLayer();
        if (layer != nullptr && !voice.isFree()
            && voice.getAge() + static_cast<int>(numFrames) > layer->getRegion().compositeTh)
            impl.handOverComposite(voice);
    });

    ModMatrix& mm = impl.resources_.getModMatrix();
    mm.beginCycle(numFrames);

    BeatClock& bc = impl.resources_.getBeatClock();
    bc.beginCycle(numFrames);

    MidiState& midiState = impl.resources_.getMidiState();

    if (impl.playheadMoved_ && bc.isPlaying()) {
        midiState.flushEvents();
        impl.genController_->resetSmoothers();
        impl.playheadMoved_ = false;
    }

    { // Clear effect busses
        ScopedTiming logger { callbackBreakdown.effects };
        for (int i = 0; i < impl.numOutputs_; ++i) {
            for (auto& bus : impl.getEffectBusesForOutput(i)) {
                if (bus)
                    bus->clearInputs(numFrames);
            }
        }
    }

    { // Main render block
        ScopedTiming logger { callbackBreakdown.renderMethod, ScopedTiming::Operation::addToDuration };

        // organdroid: the sounding voices only, in pool order (VoiceManager::forEachActive).
        impl.voiceManager_.forEachActive([&](Voice& voice) {
            if (voice.isFree())
                return;

            mm.beginVoice(voice.getId(), voice.getRegion()->getId(), voice.getTriggerEvent().value);

            const Region* region = voice.getRegion();
            ASSERT(region != nullptr);
            const auto& effectBuses = impl.getEffectBusesForOutput(region->output);

            voice.renderBlock(*tempSpan);
            for (size_t i = 0, n = effectBuses.size(); i < n; ++i) {
                if (auto& bus = effectBuses[i]) {
                    float addGain = region->getGainToEffectBus(i);
                    bus->addToInputs(*tempSpan, addGain, numFrames);
                }
            }
            callbackBreakdown.data += voice.getLastDataDuration();
            callbackBreakdown.amplitude += voice.getLastAmplitudeDuration();
            callbackBreakdown.filters += voice.getLastFilterDuration();
            callbackBreakdown.panning += voice.getLastPanningDuration();

            mm.endVoice();

            if (voice.toBeCleanedUp())
                voice.reset();
        });
    }

    { // Apply effect buses
        // -- note(jpc) there is always a "main" bus which is initially empty.
        //    without any <effect>, the signal is just going to flow through it.
        ScopedTiming logger { callbackBreakdown.effects, ScopedTiming::Operation::addToDuration };

        const int numChannels = static_cast<int>(buffer.getNumChannels());
        for (int i = 0; i < impl.numOutputs_; ++i) {
            tempMixSpan->fill(0.0f);
            const auto outputStart = numChannels == 0 ? 0 : (2 * i) % numChannels;
            auto outputSpan = buffer.getStereoSpan(outputStart);
            const auto& effectBuses = impl.getEffectBusesForOutput(i);
            for (auto& bus : effectBuses) {
                if (bus) {
                    bus->process(numFrames);
                    bus->mixOutputsTo(outputSpan, *tempMixSpan, numFrames);
                }
            }

            // Add the Mix output (fxNtomix opcodes)
            // -- note(jpc) the purpose of the Mix output is not known.
            //    perhaps it's designed as extension point for custom processing?
            //    as default behavior, it adds itself to the Main signal.
            outputSpan.add(*tempMixSpan);
        }
    }

    // Apply the master volume
    buffer.applyGain(db2mag(impl.volume_));

    // Process the metronome (debugging tool for host time info)
    constexpr bool metronomeEnabled = false;
    if (metronomeEnabled) {
        Metronome& metro = impl.resources_.getMetronome();
        metro.processAdding(
            bc.getRunningBeatNumber().data(), bc.getRunningBeatsPerBar().data(),
            buffer.getChannel(0), buffer.getChannel(1), numFrames);
    }

    // Perform any remaining modulators
    mm.endCycle();

    // Advance the clock to the end of cycle
    bc.endCycle();

    // Update sets of changed CCs
    impl.changedCCsLastCycle_ = impl.changedCCsThisCycle_;
    impl.changedCCsThisCycle_.clear();

    { // Clear events and advance midi time
        ScopedTiming logger { impl.callbackBreakdown_.dispatch, ScopedTiming::Operation::addToDuration };
        midiState.advanceTime(buffer.getNumFrames());
    }

    ASSERT(!hasNanInf(buffer.getConstSpan(0)));
    ASSERT(!hasNanInf(buffer.getConstSpan(1)));
    SFIZZ_CHECK(isReasonableAudio(buffer.getConstSpan(0)));
    SFIZZ_CHECK(isReasonableAudio(buffer.getConstSpan(1)));
}

void Synth::noteOn(int delay, int noteNumber, int velocity) noexcept
{
    const float normalizedVelocity = normalizeVelocity(velocity);
    hdNoteOn(delay, noteNumber, normalizedVelocity);
}

void Synth::hdNoteOn(int delay, int noteNumber, float normalizedVelocity) noexcept
{
    ASSERT(noteNumber < 128);
    ASSERT(noteNumber >= 0);
    Impl& impl = *impl_;
    ScopedTiming logger { impl.dispatchDuration_, ScopedTiming::Operation::addToDuration };

    if (impl.lastKeyswitchLists_[noteNumber].empty())
        impl.resources_.getMidiState().noteOnEvent(delay, noteNumber, normalizedVelocity);

    // Route this note to its console division(s) (DivisionGate.h): regions outside the
    // event's mask don't trigger. Unset (the default) = all-bits = stock behavior.
    // The stop filter (live registration) further narrows a retro attack to ONE stop.
    impl.dispatchDivMask_ = divisionConsumeMask();
    impl.dispatchStopCc_ = stopConsumeFilter();
    if (impl.dispatchStopCc_ == kStopFilterOctave) {
        // The key and its coupled rows at once (Composite.h, "Octave couplers").
        impl.octaveNoteOn(delay, noteNumber, normalizedVelocity);
        impl.dispatchDivMask_ = kDivisionAll;
        impl.dispatchStopCc_ = kStopFilterNone;
        return;
    }
    uint32_t& restrike = impl.restrikeMask_[noteNumber];
    if (impl.dispatchStopCc_ == kStopFilterRestrike) {
        // Strike again (Composite.h): only the divisions whose composite was let go, with
        // every drawn stop, but none still sounding on this key.
        impl.dispatchDivMask_ &= restrike;
        restrike &= ~impl.dispatchDivMask_;
        impl.dispatchStopCc_ = kStopFilterNone;
        impl.dispatchSounding_[0] = impl.dispatchSounding_[1] = 0;
        impl.voiceManager_.forEachActive([&](Voice& voice) {
            const TriggerEvent& event = voice.getTriggerEvent();
            const int cc = voice.getGateCc();
            if (cc >= 0 && cc < 128 && !voice.isFree() && !voice.isNoteOff()
                && event.type == TriggerEventType::NoteOn && event.number == noteNumber
                && (voice.getDivisionMask() & impl.dispatchDivMask_) != 0)
                impl.dispatchSounding_[cc >> 6] |= uint64_t { 1 } << (cc & 63);
        });
        impl.dispatchRestrike_ = true;
    } else if (impl.dispatchStopCc_ >= 0) {
        impl.dispatchDivMask_ &= ~restrike;   // the strike to come starts this stop there
    }
    if (impl.dispatchDivMask_ != 0)
        impl.noteOnDispatch(delay, noteNumber, normalizedVelocity);
    impl.dispatchRestrike_ = false;
    impl.dispatchDivMask_ = kDivisionAll;
    impl.dispatchStopCc_ = kStopFilterNone;
}

void Synth::noteOff(int delay, int noteNumber, int velocity) noexcept
{
    const float normalizedVelocity = normalizeVelocity(velocity);
    hdNoteOff(delay, noteNumber, normalizedVelocity);
}

void Synth::hdNoteOff(int delay, int noteNumber, float normalizedVelocity) noexcept
{
    ASSERT(noteNumber < 128);
    ASSERT(noteNumber >= 0);
    Impl& impl = *impl_;
    ScopedTiming logger { impl.dispatchDuration_, ScopedTiming::Operation::addToDuration };

    // FIXME: Some keyboards (e.g. Casio PX5S) can send a real note-off velocity. In this case, do we have a
    // way in sfz to specify that a release trigger should NOT use the note-on velocity?
    // auto replacedVelocity = (velocity == 0 ? getNoteVelocity(noteNumber) : velocity);
    MidiState& midiState = impl.resources_.getMidiState();

    // Division-gated release (DivisionGate.h): only voices started by this division's
    // note-on respond, so the same key held on another manual keeps sounding. The stop
    // filter (live registration) further narrows a retro release to ONE stop's voices.
    impl.dispatchDivMask_ = divisionConsumeMask();
    impl.dispatchStopCc_ = stopConsumeFilter();

    // An octave release is not a note-off of the key: the rows' own note-offs follow it.
    if (impl.lastKeyswitchLists_[noteNumber].empty() && impl.dispatchStopCc_ != kStopFilterOctave)
        midiState.noteOffEvent(delay, noteNumber, normalizedVelocity);

    const auto replacedVelocity = midiState.getNoteVelocity(noteNumber);

    // A composite sounding this key: a plain one struck on it, or an octave one with this
    // key among its rows (Composite.h, "Octave couplers").
    const auto coversNote = [&](const Voice& voice) {
        const Layer* layer = voice.getCompositeLayer();
        const TriggerEvent& event = voice.getTriggerEvent();
        if (layer == nullptr || voice.isFree() || event.type != TriggerEventType::NoteOn
            || (voice.getDivisionMask() & impl.dispatchDivMask_) == 0)
            return false;
        const int octaves = layer->getRegion().compositeOctaves;
        return octaves == 0 ? event.number == noteNumber
                            : Layer::CompositeLink::rowOf(event.number, octaves, noteNumber);
    };
    const auto isOctave = [](const Voice& voice) {
        return voice.getCompositeLayer()->getRegion().compositeOctaves != 0;
    };
    // A plain composite struck on this key.
    const auto isCompositeOnNote = [&](const Voice& voice) {
        return coversNote(voice) && !isOctave(voice);
    };

    if (impl.dispatchStopCc_ == kStopFilterOctave) {
        // Every row of the key's octave composites goes at once: each fades into its
        // summed tail, as a plain composite does. Nothing else is touched; the rows'
        // note-offs that follow find it gone, and release what the key sounds live.
        std::array<std::pair<Voice*, Layer*>, 16> tails;
        size_t numTails = 0;
        impl.voiceManager_.forEachActive([&](Voice& voice) {
            if (!coversNote(voice) || !isOctave(voice) || voice.isNoteOff()
                || voice.getTriggerEvent().number != noteNumber)
                return;
            Layer* tail = voice.getCompositeLayer()->composite_.tail;
            if (tail == nullptr)
                impl.prepareCompositeFade(voice);
            if (tail && sfz::releaseAlignEnabled() && numTails < tails.size()
                && tail->registerNoteOff(noteNumber, replacedVelocity,
                                         impl.randNoteDistribution_(Random::randomGenerator))) {
                voice.prepareCompositeRelease(sfz::releaseAlignCrossfade());
                tails[numTails++] = { &voice, tail };
            }
            voice.registerNoteOff(delay, noteNumber, replacedVelocity);
        });
        for (size_t i = 0; i < numTails; ++i)
            impl.startCompositeTail(*tails[i].first, tails[i].second, delay, replacedVelocity);
        impl.dispatchDivMask_ = kDivisionAll;
        impl.dispatchStopCc_ = kStopFilterNone;
        return;
    }

    if (impl.dispatchStopCc_ == kStopFilterCompositeRelease) {
        // A registration change before T_h (Composite.h, Restrike): each composite of this
        // key lets go into its summed tail, as a note-off would, and nothing else of the
        // key is touched; the key is marked to be struck again on its divisions (an octave
        // composite: each of its rows, which the app strikes again one by one).
        std::array<std::pair<Voice*, Layer*>, 16> tails;
        size_t numTails = 0;
        impl.voiceManager_.forEachActive([&](Voice& voice) {
            if (!coversNote(voice) || voice.isNoteOff())
                return;
            const uint32_t divisions = voice.getDivisionMask() & impl.dispatchDivMask_;
            const int key = voice.getTriggerEvent().number;
            const int octaves = voice.getCompositeLayer()->getRegion().compositeOctaves;
            for (int row : { key - 12, key, key + 12 })
                if (row >= 0 && row < 128 && (octaves == 0 ? row == noteNumber : Layer::CompositeLink::rowOf(key, octaves, row)))
                    impl.restrikeMask_[row] |= divisions;
            // on the key it was struck on, the one its tail region answers to
            Layer* tail = voice.getCompositeLayer()->composite_.tail;
            if (tail == nullptr)
                impl.prepareCompositeFade(voice);
            if (tail && sfz::releaseAlignEnabled() && numTails < tails.size()
                && tail->registerNoteOff(key, replacedVelocity,
                                         impl.randNoteDistribution_(Random::randomGenerator))) {
                voice.prepareCompositeRelease(sfz::releaseAlignCrossfade());
                tails[numTails++] = { &voice, tail };
            }
            voice.registerNoteOff(delay, key, replacedVelocity);
        });
        for (size_t i = 0; i < numTails; ++i)
            impl.startCompositeTail(*tails[i].first, tails[i].second, delay, replacedVelocity);
        impl.dispatchDivMask_ = kDivisionAll;
        impl.dispatchStopCc_ = kStopFilterNone;
        return;
    }
    if (impl.dispatchStopCc_ < 0)
        impl.restrikeMask_[noteNumber] &= ~impl.dispatchDivMask_;   // the key is let go

    // Composite stops (Composite.h): a stop retired from a held note hands the note's
    // composite over first, so that stop's own voice exists to be released below. A note-off
    // of one row of an octave composite does too, whatever it releases: the other rows may
    // still be held (by this key, or another), and only live pipes can go one row at a time.
    if (impl.dispatchStopCc_ >= 0 || impl.dispatchStopCc_ == kStopFilterNone) {
        impl.voiceManager_.forEachActive([&](Voice& voice) {
            if (!coversNote(voice) || voice.isNoteOff())
                return;
            if (impl.dispatchStopCc_ >= 0 ? voice.getCompositeLayer()->composite_.hasMemberCc(impl.dispatchStopCc_)
                                          : isOctave(voice))
                impl.handOverComposite(voice);
        });
    }

    // A composite released by this note-off fades into its summed tail, when tails are on
    // and its tail region's conditions (the tails CC) hold. Its members' own release regions
    // must not start: their tails are in the sum.
    std::array<std::pair<Voice*, Layer*>, 16> compositeTails;
    size_t numCompositeTails = 0;
    impl.dispatchSuppressCc_[0] = impl.dispatchSuppressCc_[1] = 0;
    if (impl.dispatchStopCc_ < 0) {
        impl.voiceManager_.forEachActive([&](Voice& voice) {
            if (!isCompositeOnNote(voice) || voice.isNoteOff())
                return;
            const Layer::CompositeLink& link = voice.getCompositeLayer()->composite_;
            impl.dispatchSuppressCc_[0] |= link.memberCcs[0];
            impl.dispatchSuppressCc_[1] |= link.memberCcs[1];
            if (link.tail == nullptr)
                impl.prepareCompositeFade(voice);
            Layer* tail = link.tail;
            if (tail && sfz::releaseAlignEnabled() && numCompositeTails < compositeTails.size()
                && tail->registerNoteOff(noteNumber, replacedVelocity,
                                         impl.randNoteDistribution_(Random::randomGenerator))) {
                voice.prepareCompositeRelease(sfz::releaseAlignCrossfade());
                compositeTails[numCompositeTails++] = { &voice, tail };
            }
        });
    }

    impl.voiceManager_.forEachActive([&](Voice& voice) {
        if ((voice.getDivisionMask() & impl.dispatchDivMask_) != 0
            && (impl.dispatchStopCc_ < 0 || voice.getGateCc() == impl.dispatchStopCc_))
            voice.registerNoteOff(delay, noteNumber, replacedVelocity);
    });

    for (size_t i = 0; i < numCompositeTails; ++i)
        impl.startCompositeTail(*compositeTails[i].first, compositeTails[i].second, delay, replacedVelocity);

    impl.noteOffDispatch(delay, noteNumber, replacedVelocity);
    impl.dispatchSuppressCc_[0] = impl.dispatchSuppressCc_[1] = 0;
    impl.dispatchDivMask_ = kDivisionAll;
    impl.dispatchStopCc_ = kStopFilterNone;
}

Voice* Synth::Impl::startVoice(Layer* layer, int delay, const TriggerEvent& triggerEvent, SisterVoiceRingBuilder& ring) noexcept
{
    const Region& region = layer->getRegion();

    voiceManager_.checkPolyphony(&region, delay, triggerEvent);
    Voice* selectedVoice = voiceManager_.findFreeVoice();
    if (selectedVoice == nullptr)
        return nullptr;

    selectedVoice->reset();

    // Phase-aligned release tails (ReleaseAlign.h): when this is a release voice and the
    // master enable is on, start the release sample in phase with the sustain loop it is
    // replacing. `region` here is the RELEASE region; the alignment data is keyed by the
    // ATTACK sample (the one whose live loop position we can read), so look it up via the
    // attack voice currently sounding this note.
    bool releaseAligned = false;
    if (sfz::releaseAlignEnabled() && region.isRelease() && region.compositeId < 0) {
        if (Voice* attack = voiceManager_.findAttackVoice(&region, dispatchDivMask_)) {
            float period = 0.0f; int anchor = 0; float loopRms = 0.0f;
            const uint8_t* track = nullptr; size_t trackSize = 0;
            if (sfz::releaseAlignLookup(attack->getRegion()->sampleId->filename(), period, anchor, loopRms,
                                        track, trackSize)) {
                const int loopPeriod = static_cast<int>(std::lround(period));
                const int position = attack->getSourcePosition();
                float phi = std::fmod(static_cast<float>(position - attack->getLoopStart()), period);
                if (phi < 0.0f)
                    phi += period;
                // The anchor is measured at loop_start; the phase track corrects it at the
                // attack's actual position, where the recording has drifted (RELEASE_PHASE_TRACK_DESIGN.md).
                phi += sfz::releaseAlignTrackCorrection(track, trackSize, period, position);
                int start = static_cast<int>(std::lround(anchor + phi));
                const int guard = 64;   // leave a few frames of crossfade headroom
                while (start < guard && loopPeriod > 0)
                    start += loopPeriod;
                selectedVoice->setReleaseStartOverride(start);
                // Scale the tail to the note's ACTUAL current level (GrandOrgue does this):
                // a short note on a slow-onset pipe is still quiet -- but that quietness is
                // in the SAMPLE, not the 1 ms amplitude EG -- so a full-level tail swells
                // ("ta-da"). Use the live output power vs the pipe's full (loop) level:
                //   f = currentRMS / (baseGain * loopRms)   (both in (L+R) output units).
                if (loopRms > 1e-6f) {
                    // The power follower hears every gain of the amplitude stage, but
                    // baseGain is only `amplitude`. Count the region's volume and its CC7
                    // Volume connection in the full level too: sfizz connects CC7 to every
                    // region by default (x² of 100/127 = 0.62), and without it a fully
                    // spoken pipe reads as 62 % and its tail is turned down by that much.
                    float full = attack->getBaseGain() * db2mag(attack->getBaseVolumedB()) * loopRms;
                    const Region* ar = attack->getRegion();
                    if (auto depth = ar->ccModDepth(7, ModId::Amplitude)) {
                        const auto params = ar->ccModParameters(7, ModId::Amplitude);
                        const Curve& curve = resources_.getCurves().getCurve(params->curve);
                        full *= *depth * curve.evalNormalized(resources_.getMidiState().getCCValue(7));
                    }
                    float f = (full > 1e-9f) ? std::sqrt(attack->getCurrentPower()) / full : 1.0f;
                    f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
                    selectedVoice->setReleaseGainScale(f);
                }
                releaseAligned = true;
            }
        }
    }

    if (selectedVoice->startVoice(layer, delay, triggerEvent)) {
        // Record the divisions this voice actually SOUNDS for (DivisionGate.h): the note
        // event's mask intersected with the region's own division. A coupled note-on
        // (mask = dest|src) starts each stop's voice with only that stop's division bit,
        // so a later coupler retro-release of src leaves dest's voices sounding. Untagged
        // regions (all-bits) keep the event mask = stock behavior. startVoice() reset it.
        selectedVoice->setDivisionMask(dispatchDivMask_ & region.divMask);
        // And its stop's gate CC, so a stop-filtered retro note-off releases only that
        // stop's voices (live registration).
        selectedVoice->setGateCc(region.gateCc);
        // Stretch the aligned release voice's linear fade-in to the crossfade window so it
        // is complementary to the sustain's linear fade-out (ReleaseAlign.h). Done after
        // startVoice (which resets the EG); stopDynamicUpdates() pins it past ampeg_attack.
        if (releaseAligned) {
            if (ADSREnvelope* eg = selectedVoice->getAmplitudeEG()) {
                eg->setAttackTime(sfz::releaseAlignCrossfade());
                eg->stopDynamicUpdates();
            }
        }
        ring.addVoiceToRing(selectedVoice);
        return selectedVoice;
    }
    return nullptr;
}

void Synth::Impl::resolveComposites() noexcept
{
    // Every attack region of each file: the same file can be in the palette more than once
    // (one stop loaded on two manuals), and a member must be the one on the composite's
    // own manual, or its pipes are not the ones kept silent (and the handover starts the
    // other manual's copy): those pipes would sound twice.
    absl::flat_hash_map<std::string, std::vector<Layer*>> attacks;
    absl::flat_hash_map<std::string, Layer*> tails;
    bool any = false;
    for (const LayerPtr& layerPtr : layers_) {
        const Region& region = layerPtr->getRegion();
        layerPtr->composite_ = Layer::CompositeLink {};
        if (region.isOscillator() || !region.sampleId)
            continue;
        const std::string name { region.sampleId->filename() };
        if (region.compositeId >= 0) {
            any = true;
            if (region.isRelease())
                tails.emplace(name, layerPtr.get());
        } else if (!region.isRelease()) {
            attacks[name].push_back(layerPtr.get());
        }
    }
    if (!any)
        return;

    for (const LayerPtr& layerPtr : layers_) {
        const Region& region = layerPtr->getRegion();
        if (region.compositeId < 0 || region.isRelease())
            continue;
        Layer::CompositeLink& link = layerPtr->composite_;
        bool ok = region.compositeTh > 0 && !region.compositeMembers.empty()
            && region.compositeMembers.size() <= kMaxCompositeMembers;
        // An octave composite names each member's key; a plain one sounds them all on its own.
        const bool octave = region.compositeOctaves != 0;
        ok = ok && (octave ? region.compositeMemberKeys.size() == region.compositeMembers.size()
                           : region.compositeMemberKeys.empty());
        if (octave)
            compositeOctavesById_[region.compositeId] = region.compositeOctaves;
        if (ok && !region.compositeTail.empty()) {
            const auto it = tails.find(region.compositeTail);
            ok = it != tails.end();
            if (ok)
                link.tail = it->second;
        }
        if (ok && !region.compositeStaccato.empty()) {
            const auto it = tails.find(region.compositeStaccato);
            ok = it != tails.end();
            if (ok)
                link.staccato = it->second;
        }
        for (size_t i = 0; ok && i < region.compositeMembers.size(); ++i) {
            // The member: a region of that file on the composite's manual, the one whose
            // keys hold the member's key if there are several (a sample shared by keys):
            // the composite's key, or for an octave composite the row's (its memberkeys).
            // It must carry its stop's gate CC: that is how its own pipes are kept silent
            // while the composite plays them.
            // With `composite_stops`, it must also be one of those stops: a borrowed rank
            // plays the same file under two stops of one manual.
            Layer* member = nullptr;
            const auto it = attacks.find(region.compositeMembers[i]);
            const bool byCc = (region.compositeStops[0] | region.compositeStops[1]) != 0;
            const int key = octave ? region.compositeMemberKeys[i] : region.keyRange.getStart();
            if (it != attacks.end()) {
                for (Layer* cand : it->second) {
                    const Region& r = cand->getRegion();
                    if ((r.divMask & region.divMask) == 0 || r.gateCc < 0 || r.gateCc >= 128)
                        continue;
                    if (byCc && !((region.compositeStops[r.gateCc >> 6] >> (r.gateCc & 63)) & 1u))
                        continue;
                    // One file twice in the list is two stops sharing a pipe (a borrowed
                    // bass): each entry is a different stop's region. On an octave
                    // composite, one region on two rows is one stop's sample on two keys.
                    bool taken = false;
                    for (size_t j = 0; j < link.members.size() && !taken; ++j)
                        taken = link.members[j] == cand && link.memberKeys[j] == key;
                    if (taken)
                        continue;
                    if (member == nullptr || r.keyRange.containsWithEnd(key))
                        member = cand;
                    if (r.keyRange.containsWithEnd(key))
                        break;
                }
            }
            ok = member != nullptr;
            if (ok) {
                const int cc = member->getRegion().gateCc;
                link.members.push_back(member);
                link.memberKeys.push_back(key);
                link.memberLevels.reserve(region.compositeMembers.size());
                link.memberCcs[cc >> 6] |= uint64_t { 1 } << (cc & 63);
                const int row = key < region.keyRange.getStart() ? 0 : (key > region.keyRange.getStart() ? 2 : 1);
                link.rowCcs[row][cc >> 6] |= uint64_t { 1 } << (cc & 63);
            }
        }
        link.valid = ok;
        if (!ok) {
            DBG("[sfizz] Composite region " << region.sampleId->filename() << " is not used: "
                << "its tail or a member is missing, or a member has no stop_cc");
            link = Layer::CompositeLink {};
        }
    }
}

void Synth::Impl::compositeLevels(const Layer::CompositeLink& link) noexcept
{
    // Each member's full level as the tail scale counts it (its region's amplitude and
    // volume, and its loop's rms from the release-alignment table) and the sum's, added in
    // power. Once per composite region; 0 when a member has no table entry.
    if (link.fullLevel >= 0.0f)
        return;
    double sum = 0.0;
    bool known = true;
    link.memberLevels.clear();
    for (const Layer* member : link.members) {
        const Region& r = member->getRegion();
        float period = 0.0f, loopRms = 0.0f; int anchor = 0;
        const uint8_t* track = nullptr; size_t trackSize = 0;
        if (!sfz::releaseAlignLookup(std::string(r.sampleId->filename()), period, anchor, loopRms, track, trackSize)) {
            known = false;
            break;
        }
        const double a = static_cast<double>(r.getBaseGain()) * db2mag(r.volume) * loopRms;
        link.memberLevels.push_back(static_cast<float>(a));
        sum += a * a;
    }
    if (!known)
        link.memberLevels.clear();
    link.fullLevel = known ? static_cast<float>(std::sqrt(sum)) : 0.0f;
}

void Synth::Impl::handOverComposite(Voice& composite) noexcept
{
    const Layer* layer = composite.getCompositeLayer();
    if (layer == nullptr || composite.isFree())
        return;
    const Layer::CompositeLink& link = layer->composite_;
    compositeLevels(link);
    const TriggerEvent event = composite.getTriggerEvent();
    const uint32_t divisions = composite.getDivisionMask();
    SisterVoiceRingBuilder ring;

    std::array<Voice*, kMaxCompositeMembers> started {};
    const size_t n = std::min(link.members.size(), started.size());
    for (size_t i = 0; i < n; ++i) {
        // Each on its own key: an octave composite's coupled rows are notes an octave away,
        // which their own note-offs find, as live.
        TriggerEvent own = event;
        if (i < link.memberKeys.size())
            own.number = link.memberKeys[i];
        started[i] = startVoice(link.members[i], 0, own, ring);
    }

    // Starting a voice can steal one when none is free; if that was the composite itself,
    // its state is gone and the members simply start from their own beginning.
    if (composite.getCompositeLayer() == layer) {
        const float fullSq = link.fullLevel * link.fullLevel;
        for (size_t i = 0; i < n; ++i) {
            if (started[i] == nullptr)
                continue;
            started[i]->takeOver(composite);
            const float share = (fullSq > 0.0f && i < link.memberLevels.size())
                ? link.memberLevels[i] * link.memberLevels[i] / fullSq
                : 1.0f / static_cast<float>(n);
            started[i]->takeOverPower(composite, share);
        }
        composite.reset();
    }
    for (size_t i = 0; i < n; ++i)
        if (started[i] != nullptr)
            started[i]->setDivisionMask(divisions & link.members[i]->getRegion().divMask);
}

void Synth::Impl::octaveNoteOn(int delay, int noteNumber, float velocity) noexcept
{
    const uint32_t mask = dispatchDivMask_;
    // Each division's couplers, from its active octave composite (0: none active).
    const auto couplersOf = [this](int division) {
        const int id = compositeOctaveActive(division);
        const auto it = id < 0 ? compositeOctavesById_.end() : compositeOctavesById_.find(id);
        return it == compositeOctavesById_.end() ? 0 : it->second;
    };
    // The coupled rows' own note-ons, as live (hdNoteOn counted the key's).
    int rows = 0;
    for (int d = 0; d < 32; ++d)
        if ((mask >> d) & 1u)
            rows |= couplersOf(d);
    MidiState& midiState = resources_.getMidiState();
    for (int row : { noteNumber - 12, noteNumber + 12 })
        if (row >= 0 && row < 128 && Layer::CompositeLink::rowOf(noteNumber, rows, row) && row != noteNumber
            && lastKeyswitchLists_[row].empty())
            midiState.noteOnEvent(delay, row, velocity);

    // The octave composite of each division, one draw of it, as the plain composite path
    // picks one (noteOnDispatch).
    uint32_t left = mask;
    const auto randValue = randNoteDistribution_(Random::randomGenerator);
    SisterVoiceRingBuilder ring;
    for (Layer* layer : compositeOctaveLists_[noteNumber]) {
        const Region& region = layer->getRegion();
        const Layer::CompositeLink& link = layer->composite_;
        const uint32_t divisions = region.divMask & left;
        uint32_t active = 0;
        for (int d = 0; d < 32; ++d)
            if (((divisions >> d) & 1u) && compositeOctaveActive(d) == region.compositeId)
                active |= 1u << d;
        if (!link.valid || active == 0 || !layer->registerNoteOn(noteNumber, velocity, randValue))
            continue;
        dispatchDivMask_ = active;
        const TriggerEvent triggerEvent { TriggerEventType::NoteOn, noteNumber, velocity };
        Voice* voice = startVoice(layer, delay, triggerEvent, ring);
        if (voice == nullptr)
            continue;   // no voice: the rows play as plain notes below
        voice->setCompositeLayer(layer);
        for (const Layer* member : link.members)
            voice->holdCompositeStream(member->getRegion().sampleId);
        left &= ~active;
        // The pipes of its rows that its sum left out (no recorded tail or alignment entry
        // there) play live beside it, as beside a plain composite.
        dispatchStopCc_ = kStopFilterNone;
        dispatchBesideOctave_ = true;
        const int key = noteNumber;
        int index = 0;
        for (int row : { key - 12, key, key + 12 }) {
            const uint64_t* members = link.rowCcs[index++];
            if (row < 0 || row >= 128 || !Layer::CompositeLink::rowOf(key, region.compositeOctaves, row))
                continue;
            dispatchRowMembers_[0] = members[0];
            dispatchRowMembers_[1] = members[1];
            noteOnDispatch(delay, row, velocity);
        }
        dispatchBesideOctave_ = false;
        dispatchRowMembers_[0] = dispatchRowMembers_[1] = 0;
        dispatchStopCc_ = kStopFilterOctave;
    }

    // Where none sounds (a key it has no region for: fewer than two pipes there), each row
    // as a plain note, through the plain composites like any other note.
    dispatchStopCc_ = kStopFilterNone;
    for (int couplers = 0; couplers <= 3 && left != 0; ++couplers) {
        uint32_t divisions = 0;
        for (int d = 0; d < 32; ++d)
            if (((left >> d) & 1u) && couplersOf(d) == couplers)
                divisions |= 1u << d;
        if (divisions == 0)
            continue;
        left &= ~divisions;
        for (int row : { noteNumber, noteNumber - 12, noteNumber + 12 }) {
            if (row < 0 || row >= 128 || !Layer::CompositeLink::rowOf(noteNumber, couplers, row))
                continue;
            dispatchDivMask_ = divisions;
            noteOnDispatch(delay, row, velocity);
        }
    }
    dispatchDivMask_ = mask;
}

void Synth::Impl::prepareCompositeFade(Voice& composite) noexcept
{
    // Its members share the envelope it plays with (the renderer sums no others), so it
    // already fades as they would, except in one case: with the release tails on, a looping
    // pipe that has no alignment entry fades over the fallback time instead (Voice.cpp,
    // registerNoteOff). The composite does not loop, so it is told here.
    const Layer* layer = composite.getCompositeLayer();
    if (layer == nullptr || layer->composite_.members.empty() || !sfz::releaseAlignEnabled())
        return;
    const Region& member = layer->composite_.members[0]->getRegion();
    if (!member.shouldLoop())
        return;
    float period = 0.0f, loopRms = 0.0f; int anchor = 0;
    const uint8_t* track = nullptr; size_t trackSize = 0;
    if (sfz::releaseAlignLookup(member.sampleId->filename(), period, anchor, loopRms, track, trackSize))
        return;
    if (ADSREnvelope* eg = composite.getAmplitudeEG()) {
        eg->setReleaseTime(sfz::releaseAlignFallback());
        eg->stopDynamicUpdates();
    }
}

void Synth::Impl::startCompositeTail(Voice& composite, Layer* tail, int delay, float velocity) noexcept
{
    const Layer* layer = composite.getCompositeLayer();
    if (layer == nullptr)
        return;

    startCompositeTailVoice(composite, tail, layer->getRegion().compositeTailScale, delay, velocity);
    // The staccato tail (Region::compositeStaccato), under its tail region's own conditions.
    Layer* staccato = layer->composite_.staccato;
    if (staccato && staccato->registerNoteOff(composite.getTriggerEvent().number, velocity,
                                              randNoteDistribution_(Random::randomGenerator)))
        startCompositeTailVoice(composite, staccato, layer->getRegion().compositeStaccatoScale, delay, velocity);
}

void Synth::Impl::startCompositeTailVoice(Voice& composite, Layer* tail, const std::vector<float>& table, int delay, float velocity) noexcept
{
    // The tail scale, as each pipe's tail gets it (startVoice, release alignment): the
    // note's current level against its full level, one value per power-follower step of
    // the note's age. Each member would have had its own; the renderer measured them on
    // each member's render and fitted this tail's table (and the staccato tail's) to them
    // (Region::compositeTailScale). The composite's own power cannot stand for them:
    // members at different stages of their speech give one sum, and its volume= holds the
    // file's storage gain. A tail at zero is not started.
    float scale = 1.0f;
    if (!table.empty()) {
        const size_t k = static_cast<size_t>(std::max(composite.getAge(), 0)) / config::powerFollowerStep;
        scale = table[std::min(k, table.size() - 1)];
        scale = scale < 0.0f ? 0.0f : (scale > 1.0f ? 1.0f : scale);
        if (scale <= 0.0f)
            return;
    }

    const Region& region = tail->getRegion();
    const TriggerEvent event { TriggerEventType::NoteOff, composite.getTriggerEvent().number, velocity };
    voiceManager_.checkPolyphony(&region, delay, event);
    Voice* voice = voiceManager_.findFreeVoice();
    if (voice == nullptr)
        return;
    voice->reset();
    voice->setReleaseGainScale(scale);
    if (!voice->startVoice(tail, delay, event))
        return;
    voice->setDivisionMask(composite.getDivisionMask() & region.divMask);
    voice->setGateCc(region.gateCc);
    if (ADSREnvelope* eg = voice->getAmplitudeEG()) {
        eg->setAttackTime(sfz::releaseAlignCrossfade());
        eg->setEqualPower(true);
        eg->stopDynamicUpdates();
    }
    SisterVoiceRingBuilder ring;
    ring.addVoiceToRing(voice);
}

void Synth::Impl::checkOffGroups(const Region* region, int delay, int number, bool chokedByCC)
{
    voiceManager_.forEachActive([&](Voice& voice) {
        if (voice.checkOffGroup(region, delay, number)) {
            const TriggerEvent& event = voice.getTriggerEvent();
            if (event.type == TriggerEventType::NoteOn && !chokedByCC)
                noteOffDispatch(delay, event.number, event.value);
        }
    });
}

void Synth::Impl::noteOffDispatch(int delay, int noteNumber, float velocity) noexcept
{
    const auto randValue = randNoteDistribution_(Random::randomGenerator);
    SisterVoiceRingBuilder ring;
    const TriggerEvent triggerEvent { TriggerEventType::NoteOff, noteNumber, velocity };

    for (Layer* layer : upKeyswitchLists_[noteNumber])
        layer->keySwitched_ = true;

    for (Layer* layer : downKeyswitchLists_[noteNumber])
        layer->keySwitched_ = false;

    for (Layer* layer : noteActivationLists_[noteNumber]) {
        const Region& region = layer->getRegion();
        if ((region.divMask & dispatchDivMask_) == 0)   // DivisionGate.h
            continue;
        if (dispatchStopCc_ >= 0 && region.gateCc != dispatchStopCc_)   // live registration
            continue;
        if (region.gateCc >= 0 && ((dispatchSuppressCc_[region.gateCc >> 6] >> (region.gateCc & 63)) & 1u))
            continue;   // a composite member: its tail is in the summed tail (Composite.h)
        if (layer->registerNoteOff(noteNumber, velocity, randValue)) {
            if (region.trigger == Trigger::release && !region.rtDead && !voiceManager_.playingAttackVoice(&region))
                continue;

            checkOffGroups(&region, delay, noteNumber);
            startVoice(layer, delay, triggerEvent, ring);
        }
    }
}

void Synth::Impl::noteOnDispatch(int delay, int noteNumber, float velocity) noexcept
{
    const auto randValue = randNoteDistribution_(Random::randomGenerator);
    SisterVoiceRingBuilder ring;
    MidiState& midiState = resources_.getMidiState();

    if (!lastKeyswitchLists_[noteNumber].empty()) {
        if (currentSwitch_ && *currentSwitch_ != noteNumber) {
            for (Layer* layer : lastKeyswitchLists_[*currentSwitch_])
                layer->keySwitched_ = false;
        }
        currentSwitch_ = noteNumber;
    }

    for (Layer* layer : lastKeyswitchLists_[noteNumber])
        layer->keySwitched_ = true;

    for (Layer* layer : upKeyswitchLists_[noteNumber])
        layer->keySwitched_ = false;

    for (Layer* layer : downKeyswitchLists_[noteNumber])
        layer->keySwitched_ = true;

    // Composite stops (Composite.h): where the app has activated a composite for one of
    // this note's divisions, one composite voice (a random draw among its regions for the
    // key) plays instead of its member pipes, which are skipped below. It requests their
    // streams now, so they are read as early as in live playing. A retro attack (stop
    // filter) is a single stop's pipe and never a composite.
    uint64_t suppress[2] { dispatchRowMembers_[0], dispatchRowMembers_[1] };
    if (dispatchStopCc_ < 0 && !dispatchBesideOctave_ && compositeAnyActive()) {
        for (Layer* layer : compositeLists_[noteNumber]) {
            const Region& region = layer->getRegion();
            const Layer::CompositeLink& link = layer->composite_;
            const uint32_t divisions = region.divMask & dispatchDivMask_;
            if (!link.valid || divisions == 0)
                continue;
            bool active = false;
            for (int d = 0; d < 32 && !active; ++d)
                active = ((divisions >> d) & 1u) && compositeActive(d) == region.compositeId;
            if (dispatchRestrike_ && ((link.memberCcs[0] & dispatchSounding_[0])
                                      | (link.memberCcs[1] & dispatchSounding_[1])) != 0)
                continue;   // a member still sounds on the key: the members play live
            if (!active || !layer->registerNoteOn(noteNumber, velocity, randValue))
                continue;
            const TriggerEvent triggerEvent { TriggerEventType::NoteOn, noteNumber, velocity };
            Voice* voice = startVoice(layer, delay, triggerEvent, ring);
            if (voice == nullptr)
                continue;   // no voice: the members play live
            voice->setCompositeLayer(layer);
            for (const Layer* member : link.members)
                voice->holdCompositeStream(member->getRegion().sampleId);
            suppress[0] |= link.memberCcs[0];
            suppress[1] |= link.memberCcs[1];
        }
    }

    for (Layer* layer : noteActivationLists_[noteNumber]) {
        if ((layer->getRegion().divMask & dispatchDivMask_) == 0)   // DivisionGate.h
            continue;
        if (dispatchStopCc_ >= 0 && layer->getRegion().gateCc != dispatchStopCc_)   // live registration
            continue;
        const int gateCc = layer->getRegion().gateCc;
        if (gateCc >= 0 && ((suppress[gateCc >> 6] >> (gateCc & 63)) & 1u))
            continue;   // played by the composite
        if (dispatchRestrike_ && gateCc >= 0 && gateCc < 128
            && ((dispatchSounding_[gateCc >> 6] >> (gateCc & 63)) & 1u))
            continue;   // still sounding on the key (Composite.h, Restrike)
        if (layer->registerNoteOn(noteNumber, velocity, randValue)) {
            const Region& region = layer->getRegion();
            if (region.useTimerRange && !voiceManager_.withinValidTimerRange(&region, midiState.getInternalClock() + delay, sampleRate_))
                continue;

            checkOffGroups(&region, delay, noteNumber);
            TriggerEvent triggerEvent { TriggerEventType::NoteOn, noteNumber, velocity };
            startVoice(layer, delay, triggerEvent, ring);
        }
    }

    for (Layer* layer : previousKeyswitchLists_) {
        const Region& region = layer->getRegion();
        layer->previousKeySwitched_ = (region.previousKeyswitch == noteNumber);
    }
}

void Synth::Impl::startDelayedSustainReleases(Layer* layer, int delay, SisterVoiceRingBuilder& ring) noexcept
{
    const Region& region = layer->getRegion();

    if (!region.rtDead && !voiceManager_.playingAttackVoice(&region)) {
        layer->delayedSustainReleases_.clear();
        return;
    }

    for (auto& note: layer->delayedSustainReleases_) {
        const TriggerEvent noteOffEvent { TriggerEventType::NoteOff, note.first, note.second };
        startVoice(layer, delay, noteOffEvent, ring);
    }

    layer->delayedSustainReleases_.clear();
}

void Synth::Impl::startDelayedSostenutoReleases(Layer* layer, int delay, SisterVoiceRingBuilder& ring) noexcept
{
    const Region& region = layer->getRegion();

    if (!region.rtDead && !voiceManager_.playingAttackVoice(&region)) {
        layer->delayedSostenutoReleases_.clear();
        return;
    }

    for (auto& note: layer->delayedSostenutoReleases_) {
        const TriggerEvent noteOffEvent { TriggerEventType::NoteOff, note.first, note.second };
        startVoice(layer, delay, noteOffEvent, ring);
    }
    layer->delayedSostenutoReleases_.clear();
}

void Synth::cc(int delay, int ccNumber, int ccValue) noexcept
{
    const auto normalizedCC = normalizeCC(ccValue);
    hdcc(delay, ccNumber, normalizedCC);
}

void Synth::Impl::ccDispatch(int delay, int ccNumber, float value, int extendedArg) noexcept
{
    SisterVoiceRingBuilder ring;
    TriggerEvent triggerEvent { TriggerEventType::CC, ccNumber, value };
    const auto randValue = randNoteDistribution_(Random::randomGenerator);
    MidiState& midiState = resources_.getMidiState();
    for (Layer* layer : ccActivationLists_[ccNumber]) {
        const Region& region = layer->getRegion();

        if (region.checkSustain && ccNumber == region.sustainCC && value < region.sustainThreshold)
            startDelayedSustainReleases(layer, delay, ring);

        if (region.checkSostenuto && ccNumber == region.sostenutoCC && value < region.sostenutoThreshold) {
            if (layer->sustainPressed_) {
                for (const auto& v: layer->delayedSostenutoReleases_)
                    layer->delaySustainRelease(v.first, v.second);

                layer->delayedSostenutoReleases_.clear();
            } else {
                startDelayedSostenutoReleases(layer, delay, ring);
            }
        }

        if (layer->registerCC(ccNumber, value, randValue, extendedArg)) {
            if (region.useTimerRange && ! voiceManager_.withinValidTimerRange(&region, midiState.getInternalClock() + delay, sampleRate_))
                continue;

            checkOffGroups(&region, delay, ccNumber, true);
            startVoice(layer, delay, triggerEvent, ring);
        }
    }
}

void Synth::hdcc(int delay, int ccNumber, float normValue) noexcept
{
    Impl& impl = *impl_;
    impl.performHdcc(delay, ccNumber, normValue, true);
}

void Synth::automateHdcc(int delay, int ccNumber, float normValue) noexcept
{
    Impl& impl = *impl_;
    impl.performHdcc(delay, ccNumber, normValue, false);
}

void Synth::Impl::performHdcc(int delay, int ccNumber, float normValue, bool asMidi, int extendedArg) noexcept
{
    ASSERT(ccNumber < config::numCCs);
    ASSERT(ccNumber >= 0);

    ScopedTiming logger { dispatchDuration_, ScopedTiming::Operation::addToDuration };

    changedCCsThisCycle_.set(ccNumber);

    MidiState& midiState = resources_.getMidiState();

    if (asMidi) {
        if (ccNumber == config::resetCC) {
            resetAllControllers(delay);
            return;
        }

        if (ccNumber == config::allNotesOffCC || ccNumber == config::allSoundOffCC) {
            for (auto& voice : voiceManager_)
                voice.reset();
            midiState.allNotesOff(delay);
            return;
        }
    }

    voiceManager_.forEachActive([&](Voice& voice) { voice.registerCC(delay, ccNumber, normValue); });

    ccDispatch(delay, ccNumber, normValue, extendedArg);
    midiState.ccEvent(delay, ccNumber, normValue);
}

void Synth::Impl::setDefaultHdcc(int ccNumber, float value)
{
    ASSERT(ccNumber >= 0);
    ASSERT(ccNumber < config::numCCs);
    defaultCCValues_[ccNumber] = value;
}

float Synth::getHdcc(int ccNumber)
{
    ASSERT(ccNumber >= 0);
    ASSERT(ccNumber < config::numCCs);
    Impl& impl = *impl_;
    return impl.resources_.getMidiState().getCCValue(ccNumber);
}

float Synth::getDefaultHdcc(int ccNumber)
{
    ASSERT(ccNumber >= 0);
    ASSERT(ccNumber < config::numCCs);
    Impl& impl = *impl_;
    return impl.defaultCCValues_[ccNumber];
}

void Synth::pitchWheel(int delay, int pitch) noexcept
{
    const float normalizedPitch = normalizeBend(float(pitch));
    hdPitchWheel(delay, normalizedPitch);
}

void Synth::hdPitchWheel(int delay, float normalizedPitch) noexcept
{
    Impl& impl = *impl_;

    ScopedTiming logger { impl.dispatchDuration_, ScopedTiming::Operation::addToDuration };
    impl.resources_.getMidiState().pitchBendEvent(delay, normalizedPitch);

    for (const Impl::LayerPtr& layer : impl.layers_) {
        layer->registerPitchWheel(normalizedPitch);
    }

    impl.voiceManager_.forEachActive([&](Voice& voice) { voice.registerPitchWheel(delay, normalizedPitch); });

    impl.performHdcc(delay, ExtendedCCs::pitchBend, normalizedPitch, false);
}

void Synth::programChange(int delay, int program) noexcept
{
    Impl& impl = *impl_;
    impl.resources_.getMidiState().programChangeEvent(delay, program);
    for (const Impl::LayerPtr& layer : impl.layers_)
        layer->registerProgramChange(program);
}


void Synth::channelAftertouch(int delay, int aftertouch) noexcept
{
    const float normalizedAftertouch = normalize7Bits(aftertouch);
    hdChannelAftertouch(delay, normalizedAftertouch);
}

void Synth::hdChannelAftertouch(int delay, float normAftertouch) noexcept
{
    Impl& impl = *impl_;
    ScopedTiming logger { impl.dispatchDuration_, ScopedTiming::Operation::addToDuration };

    impl.resources_.getMidiState().channelAftertouchEvent(delay, normAftertouch);

    for (const Impl::LayerPtr& layerPtr : impl.layers_) {
        layerPtr->registerAftertouch(normAftertouch);
    }

    impl.voiceManager_.forEachActive([&](Voice& voice) { voice.registerAftertouch(delay, normAftertouch); });

    impl.performHdcc(delay, ExtendedCCs::channelAftertouch, normAftertouch, false);
}

void Synth::polyAftertouch(int delay, int noteNumber, int aftertouch) noexcept
{
    const float normalizedAftertouch = normalize7Bits(aftertouch);
    hdPolyAftertouch(delay, noteNumber, normalizedAftertouch);
}

void Synth::hdPolyAftertouch(int delay, int noteNumber, float normAftertouch) noexcept
{
    Impl& impl = *impl_;
    ScopedTiming logger { impl.dispatchDuration_, ScopedTiming::Operation::addToDuration };

    impl.resources_.getMidiState().polyAftertouchEvent(delay, noteNumber, normAftertouch);

    impl.voiceManager_.forEachActive([&](Voice& voice) { voice.registerPolyAftertouch(delay, noteNumber, normAftertouch); });

    impl.performHdcc(delay, ExtendedCCs::polyphonicAftertouch, normAftertouch, false, noteNumber);
}

void Synth::tempo(int delay, float secondsPerBeat) noexcept
{
    Impl& impl = *impl_;
    ScopedTiming logger { impl.dispatchDuration_, ScopedTiming::Operation::addToDuration };

    impl.resources_.getBeatClock().setTempo(delay, secondsPerBeat);
}

void Synth::bpmTempo(int delay, float beatsPerMinute) noexcept
{
    // TODO make this the main tempo function and remove the deprecated other one
    return tempo(delay, 60 / beatsPerMinute);
}

void Synth::timeSignature(int delay, int beatsPerBar, int beatUnit)
{
    Impl& impl = *impl_;
    ScopedTiming logger { impl.dispatchDuration_, ScopedTiming::Operation::addToDuration };

    impl.resources_.getBeatClock().setTimeSignature(delay, TimeSignature(beatsPerBar, beatUnit));
}

void Synth::timePosition(int delay, int bar, double barBeat)
{
    Impl& impl = *impl_;
    ScopedTiming logger { impl.dispatchDuration_, ScopedTiming::Operation::addToDuration };

    BeatClock& beatClock = impl.resources_.getBeatClock();

    const auto newPosition = BBT(bar, barBeat);
    const auto newBeatPosition = newPosition.toBeats(beatClock.getTimeSignature());
    const auto currentBeatPosition = beatClock.getLastBeatPosition();
    const auto positionDifference = std::abs(newBeatPosition - currentBeatPosition);
    const auto threshold = config::playheadMovedFrames * beatClock.getBeatsPerFrame();

    if (positionDifference > threshold)
        impl.playheadMoved_ = true;

    beatClock.setTimePosition(delay, newPosition);
}

void Synth::playbackState(int delay, int playbackState)
{
    Impl& impl = *impl_;
    ScopedTiming logger { impl.dispatchDuration_, ScopedTiming::Operation::addToDuration };

    impl.resources_.getBeatClock().setPlaying(delay, playbackState == 1);
}

int Synth::getNumRegions() const noexcept
{
    Impl& impl = *impl_;
    return static_cast<int>(impl.layers_.size());
}

int Synth::getNumGroups() const noexcept
{
    Impl& impl = *impl_;
    return impl.numGroups_;
}

int Synth::getNumMasters() const noexcept
{
    Impl& impl = *impl_;
    return impl.numMasters_;
}

int Synth::getNumCurves() const noexcept
{
    Impl& impl = *impl_;
    return static_cast<int>(impl.resources_.getCurves().getNumCurves());
}

std::string Synth::exportMidnam(absl::string_view model) const
{
    Impl& impl = *impl_;
    pugi::xml_document doc;
    absl::string_view manufacturer = config::midnamManufacturer;

    if (model.empty())
        model = config::midnamModel;

    doc.append_child(pugi::node_doctype).set_value("MIDINameDocument PUBLIC"
                                                   " \"-//MIDI Manufacturers Association//DTD MIDINameDocument 1.0//EN\""
                                                   " \"http://www.midi.org/dtds/MIDINameDocument10.dtd\"");

    pugi::xml_node root = doc.append_child("MIDINameDocument");

    root.append_child(pugi::node_comment)
        .set_value("Generated by Sfizz for the current instrument");

    root.append_child("Author");

    pugi::xml_node device = root.append_child("MasterDeviceNames");
    device.append_child("Manufacturer")
        .append_child(pugi::node_pcdata)
        .set_value(std::string(manufacturer).c_str());
    device.append_child("Model")
        .append_child(pugi::node_pcdata)
        .set_value(std::string(model).c_str());

    {
        pugi::xml_node devmode = device.append_child("CustomDeviceMode");
        devmode.append_attribute("Name").set_value("Default");

        pugi::xml_node nsas = devmode.append_child("ChannelNameSetAssignments");
        for (unsigned c = 0; c < 16; ++c) {
            pugi::xml_node nsa = nsas.append_child("ChannelNameSetAssign");
            nsa.append_attribute("Channel").set_value(std::to_string(c + 1).c_str());
            nsa.append_attribute("NameSet").set_value("Play");
        }
    }

    {
        pugi::xml_node chns = device.append_child("ChannelNameSet");
        chns.append_attribute("Name").set_value("Play");

        pugi::xml_node acs = chns.append_child("AvailableForChannels");
        for (unsigned c = 0; c < 16; ++c) {
            pugi::xml_node ac = acs.append_child("AvailableChannel");
            ac.append_attribute("Channel").set_value(std::to_string(c + 1).c_str());
            ac.append_attribute("Available").set_value("true");
        }

        chns.append_child("UsesControlNameList")
            .append_attribute("Name")
            .set_value("Controls");
        chns.append_child("UsesNoteNameList")
            .append_attribute("Name")
            .set_value("Notes");
    }

    {
        auto anonymousCCs = getUsedCCs();

        pugi::xml_node cns = device.append_child("ControlNameList");
        cns.append_attribute("Name").set_value("Controls");
        for (const auto& pair : impl.ccLabels_) {
            anonymousCCs.set(pair.first, false);
            if (pair.first < 128) {
                pugi::xml_node cn = cns.append_child("Control");
                cn.append_attribute("Type").set_value("7bit");
                cn.append_attribute("Number").set_value(std::to_string(pair.first).c_str());
                cn.append_attribute("Name").set_value(pair.second.c_str());
            }
        }

        for (unsigned i = 0, n = std::min<unsigned>(128, anonymousCCs.bit_size()); i < n; ++i) {
            if (anonymousCCs.test(i)) {
                pugi::xml_node cn = cns.append_child("Control");
                cn.append_attribute("Type").set_value("7bit");
                cn.append_attribute("Number").set_value(std::to_string(i).c_str());
                cn.append_attribute("Name").set_value(("Unnamed CC " + std::to_string(i)).c_str());
            }
        }
    }

    {
        pugi::xml_node nnl = device.append_child("NoteNameList");
        nnl.append_attribute("Name").set_value("Notes");
        for (const auto& pair : impl.keyswitchLabels_) {
            pugi::xml_node nn = nnl.append_child("Note");
            nn.append_attribute("Number").set_value(std::to_string(pair.first).c_str());
            nn.append_attribute("Name").set_value(pair.second.c_str());
        }
        for (const auto& pair : impl.keyLabels_) {
            pugi::xml_node nn = nnl.append_child("Note");
            nn.append_attribute("Number").set_value(std::to_string(pair.first).c_str());
            nn.append_attribute("Name").set_value(pair.second.c_str());
        }
    }

    string_xml_writer writer;
    doc.save(writer);
    return std::move(writer.str());
}

const Layer* Synth::getLayerView(int idx) const noexcept
{
    Impl& impl = *impl_;
    return (size_t)idx < impl.layers_.size() ? impl.layers_[idx].get() : nullptr;
}

const Region* Synth::getRegionView(int idx) const noexcept
{
    const Layer* layer = getLayerView(idx);
    return layer ? &layer->getRegion() : nullptr;
}

const EffectBus* Synth::getEffectBusView(int idx, int output) const noexcept
{
    Impl& impl = *impl_;
    return (size_t)idx < impl.effectBuses_[output].size() ? impl.effectBuses_[output][idx].get() : nullptr;
}

const RegionSet* Synth::getRegionSetView(int idx) const noexcept
{
    Impl& impl = *impl_;
    return (size_t)idx < impl.sets_.size() ? impl.sets_[idx].get() : nullptr;
}

const PolyphonyGroup* Synth::getPolyphonyGroupView(int idx) const noexcept
{
    Impl& impl = *impl_;
    return impl.voiceManager_.getPolyphonyGroupView(idx);
}

Layer* Synth::getLayerById(NumericId<Region> id) noexcept
{
    Impl& impl = *impl_;
    const size_t size = impl.layers_.size();

    if (size == 0 || !id.valid())
        return nullptr;

    // search a sequence of ordered identifiers with potential gaps
    size_t index = static_cast<size_t>(id.number());
    index = std::min(index, size - 1);

    while (index > 0 && impl.layers_[index]->getRegion().getId().number() > id.number())
        --index;

    return (impl.layers_[index]->getRegion().getId() == id) ?
        impl.layers_[index].get() : nullptr;
}

const Region* Synth::getRegionById(NumericId<Region> id) const noexcept
{
    Layer* layer = const_cast<Synth*>(this)->getLayerById(id);
    return layer ? &layer->getRegion() : nullptr;
}

const Voice* Synth::getVoiceView(int idx) const noexcept
{
    Impl& impl = *impl_;
    return idx < impl.numVoices_ ? &impl.voiceManager_[idx] : nullptr;
}

unsigned Synth::getNumPolyphonyGroups() const noexcept
{
    Impl& impl = *impl_;
    return impl.voiceManager_.getNumPolyphonyGroups();
}

const std::vector<std::string>& Synth::getUnknownOpcodes() const noexcept
{
    Impl& impl = *impl_;
    return impl.unknownOpcodes_;
}
size_t Synth::getNumPreloadedSamples() const noexcept
{
    Impl& impl = *impl_;
    return impl.resources_.getFilePool().getNumPreloadedSamples();
}

int Synth::getSampleQuality(ProcessMode mode)
{
    Impl& impl = *impl_;
    SynthConfig& synthConfig = impl.resources_.getSynthConfig();
    switch (mode) {
    case ProcessLive:
        return synthConfig.liveSampleQuality;
    case ProcessFreewheeling:
        return synthConfig.freeWheelingSampleQuality;
    default:
        SFIZZ_CHECK(false);
        return 0;
    }
}

void Synth::setSampleQuality(ProcessMode mode, int quality)
{
    SFIZZ_CHECK(quality >= 0 && quality <= 10);
    Impl& impl = *impl_;
    quality = clamp(quality, 0, 10);
    SynthConfig& synthConfig = impl.resources_.getSynthConfig();

    switch (mode) {
    case ProcessLive:
        synthConfig.liveSampleQuality = quality;
        break;
    case ProcessFreewheeling:
        // DBG("Set freewheeling quality" << quality);
        synthConfig.freeWheelingSampleQuality = quality;
        break;
    default:
        SFIZZ_CHECK(false);
        break;
    }
}

int Synth::getOscillatorQuality(ProcessMode mode)
{
    Impl& impl = *impl_;
    SynthConfig& synthConfig = impl.resources_.getSynthConfig();
    switch (mode) {
    case ProcessLive:
        return synthConfig.liveOscillatorQuality;
    case ProcessFreewheeling:
        return synthConfig.freeWheelingOscillatorQuality;
    default:
        SFIZZ_CHECK(false);
        return 0;
    }
}

void Synth::setOscillatorQuality(ProcessMode mode, int quality)
{
    SFIZZ_CHECK(quality >= 0 && quality <= 3);
    Impl& impl = *impl_;
    quality = clamp(quality, 0, 3);
    SynthConfig& synthConfig = impl.resources_.getSynthConfig();

    switch (mode) {
    case ProcessLive:
        synthConfig.liveOscillatorQuality = quality;
        break;
    case ProcessFreewheeling:
        // DBG("Set freewheeling oscillator quality" << quality);
        synthConfig.freeWheelingOscillatorQuality = quality;
        break;
    default:
        SFIZZ_CHECK(false);
        break;
    }
}

void Synth::setSustainCancelsRelease(bool value)
{
    impl_->resources_.getSynthConfig().sustainCancelsRelease = value;
}

float Synth::getVolume() const noexcept
{
    Impl& impl = *impl_;
    return impl.volume_;
}
void Synth::setVolume(float volume) noexcept
{
    Impl& impl = *impl_;
    impl.volume_ = Default::volume.bounds.clamp(volume);
}

int Synth::getNumVoices() const noexcept
{
    Impl& impl = *impl_;
    return impl.numVoices_;
}

void Synth::setNumVoices(int numVoices) noexcept
{
    ASSERT(numVoices > 0);
    Impl& impl = *impl_;

    // fast path
    if (numVoices == impl.numVoices_)
        return;

    impl.resetVoices(numVoices);
}

void Synth::Impl::resetVoices(int numVoices)
{
    numVoices_ = numVoices;

    for (auto& set : sets_)
        set->removeAllVoices();

    voiceManager_.requireNumVoices(numVoices_, resources_);

    for (auto& voice : voiceManager_) {
        voice.setSampleRate(this->sampleRate_);
        voice.setSamplesPerBlock(this->samplesPerBlock_);
    }

    applySettingsPerVoice();
}

void Synth::Impl::resetCallbackBreakdown()
{
    callbackBreakdown_ = CallbackBreakdown();
}

void Synth::Impl::applySettingsPerVoice()
{
    for (auto& voice : voiceManager_) {
        voice.setMaxFiltersPerVoice(settingsPerVoice_.maxFilters);
        voice.setMaxEQsPerVoice(settingsPerVoice_.maxEQs);
        voice.setMaxLFOsPerVoice(settingsPerVoice_.maxLFOs);
        voice.setMaxFlexEGsPerVoice(settingsPerVoice_.maxFlexEGs);
        voice.setPitchEGEnabledPerVoice(settingsPerVoice_.havePitchEG);
        voice.setFilterEGEnabledPerVoice(settingsPerVoice_.haveFilterEG);
        voice.setAmplitudeLFOEnabledPerVoice(settingsPerVoice_.haveAmplitudeLFO);
        voice.setPitchLFOEnabledPerVoice(settingsPerVoice_.havePitchLFO);
        voice.setFilterLFOEnabledPerVoice(settingsPerVoice_.haveFilterLFO);
    }
}

void Synth::Impl::setupModMatrix()
{
    ModMatrix& mm = resources_.getModMatrix();

    for (const LayerPtr& layerPtr : layers_) {
        const Region& region = layerPtr->getRegion();

        for (const Region::Connection& conn : region.connections) {
            ModGenerator* gen = nullptr;

            ModKey sourceKey = conn.source;
            ModKey targetKey = conn.target;

            // normalize the stepcc to 0-1
            if (sourceKey.id() == ModId::Controller) {
                ModKey::Parameters p = sourceKey.parameters();
                p.step = (conn.sourceDepth <= 0.0f) ? 0.0f :
                    (p.step / conn.sourceDepth);
                sourceKey = ModKey::createCC(p.cc, p.curve, p.smooth, p.step);
            }

            switch (sourceKey.id()) {
            case ModId::Controller:
            case ModId::PerVoiceController:
                gen = genController_.get();
                break;
            case ModId::AmpLFO:
            case ModId::PitchLFO:
            case ModId::FilLFO:
            case ModId::LFO:
                gen = genLFO_.get();
                break;
            case ModId::Envelope:
                gen = genFlexEnvelope_.get();
                break;
            case ModId::AmpEG:
            case ModId::PitchEG:
            case ModId::FilEG:
                gen = genADSREnvelope_.get();
                break;
            case ModId::ChannelAftertouch:
                gen = genChannelAftertouch_.get();
                break;
            case ModId::PolyAftertouch:
                gen = genPolyAftertouch_.get();
                break;
            default:
                DBG("[sfizz] Have unknown type of source generator");
                break;
            }

            ASSERT(gen);
            if (!gen)
                continue;

            ModMatrix::SourceId source = mm.registerSource(sourceKey, *gen);
            ModMatrix::TargetId target = mm.registerTarget(targetKey);

            ASSERT(source);
            if (!source) {
                DBG("[sfizz] Failed to register modulation source");
                continue;
            }

            ASSERT(target);
            if (!target) {
                DBG("[sfizz] Failed to register modulation target");
                continue;
            }

            if (!mm.connect(source, target, conn.sourceDepth, conn.sourceDepthMod, conn.velToDepth)) {
                DBG("[sfizz] Failed to connect modulation source and target");
                ASSERTFALSE;
            }
        }
    }

    mm.init();
}

void Synth::setPreloadSize(uint32_t preloadSize) noexcept
{
    Impl& impl = *impl_;
    FilePool& filePool = impl.resources_.getFilePool();

    // fast path
    if (preloadSize == filePool.getPreloadSize())
        return;

    filePool.setPreloadSize(preloadSize);
}

uint32_t Synth::getPreloadSize() const noexcept
{
    Impl& impl = *impl_;
    return impl.resources_.getFilePool().getPreloadSize();
}

void Synth::enableFreeWheeling() noexcept
{
    Impl& impl = *impl_;
    SynthConfig& synthConfig = impl.resources_.getSynthConfig();
    if (!synthConfig.freeWheeling) {
        synthConfig.freeWheeling = true;
        DBG("Enabling freewheeling");
    }
}
void Synth::disableFreeWheeling() noexcept
{
    Impl& impl = *impl_;
    SynthConfig& synthConfig = impl.resources_.getSynthConfig();
    if (synthConfig.freeWheeling) {
        synthConfig.freeWheeling = false;
        DBG("Disabling freewheeling");
    }
}

void Synth::Impl::resetAllControllers(int delay) noexcept
{
    MidiState& midiState = resources_.getMidiState();
    midiState.pitchBendEvent(delay, 0.0f);
    for (int cc = 0; cc < config::numCCs; ++cc)
        midiState.ccEvent(delay, cc, defaultCCValues_[cc]);

    for (auto& voice : voiceManager_) {
        voice.registerPitchWheel(delay, 0);
        for (int cc = 0; cc < config::numCCs; ++cc)
            voice.registerCC(delay, cc, defaultCCValues_[cc]);
    }

    for (const LayerPtr& layerPtr : layers_) {
        Layer& layer = *layerPtr;
        for (int cc = 0; cc < config::numCCs; ++cc)
            layer.updateCCState(cc, defaultCCValues_[cc]);
    }
}

absl::optional<fs::file_time_type> Synth::Impl::checkModificationTime() const
{
    absl::optional<fs::file_time_type> resultTime;
    for (const auto& file : parser_.getIncludedFiles()) {
        std::error_code ec;
        const auto fileTime = fs::last_write_time(file, ec);
        if (!ec) {
            if (!resultTime || fileTime > *resultTime)
                resultTime = fileTime;
        }
    }
    return resultTime;
}

bool Synth::shouldReloadFile()
{
    Impl& impl = *impl_;

    absl::optional<fs::file_time_type> then = impl.modificationTime_;
    if (!then) // file not loaded or failed
        return false;

    absl::optional<fs::file_time_type> now = impl.checkModificationTime();
    if (!now) // file not currently existing
        return false;

    return *now > *then;
}

bool Synth::shouldReloadScala()
{
    Impl& impl = *impl_;
    return impl.resources_.getTuning().shouldReloadScala();
}

const Synth::CallbackBreakdown& Synth::getCallbackBreakdown() const noexcept
{
    Impl& impl = *impl_;
    return impl.callbackBreakdown_;
}


void Synth::allSoundOff() noexcept
{
    Impl& impl = *impl_;
    for (auto& voice : impl.voiceManager_)
        voice.reset();
    impl.restrikeMask_.fill(0);
    for (int i = 0; i < impl.numOutputs_; ++i) {
        for (auto& effectBus : impl.getEffectBusesForOutput(i))
            if (effectBus)
                effectBus->clear();
    }
}

void Synth::addExternalDefinition(const std::string& id, const std::string& value)
{
    Impl& impl = *impl_;
    impl.parser_.addExternalDefinition(id, value);
}

void Synth::clearExternalDefinitions()
{
    Impl& impl = *impl_;
    impl.parser_.clearExternalDefinitions();
}

const BitArray<config::numCCs>& Synth::getUsedCCs() const noexcept
{
    Impl& impl = *impl_;
    return impl.currentUsedCCs_;
}

void sfz::Synth::setBroadcastCallback(sfizz_receive_t* broadcast, void* data)
{
    Impl& impl = *impl_;
    impl.broadcastReceiver = broadcast;
    impl.broadcastData = data;
}

void Synth::Impl::collectUsedCCsFromRegion(BitArray<config::numCCs>& usedCCs, const Region& region)
{
    collectUsedCCsFromCCMap(usedCCs, region.delayCC);
    collectUsedCCsFromCCMap(usedCCs, region.offsetCC);
    collectUsedCCsFromCCMap(usedCCs, region.endCC);
    collectUsedCCsFromCCMap(usedCCs, region.loopStartCC);
    collectUsedCCsFromCCMap(usedCCs, region.loopEndCC);
    collectUsedCCsFromCCMap(usedCCs, region.amplitudeEG.ccAttack);
    collectUsedCCsFromCCMap(usedCCs, region.amplitudeEG.ccRelease);
    collectUsedCCsFromCCMap(usedCCs, region.amplitudeEG.ccDecay);
    collectUsedCCsFromCCMap(usedCCs, region.amplitudeEG.ccDelay);
    collectUsedCCsFromCCMap(usedCCs, region.amplitudeEG.ccHold);
    collectUsedCCsFromCCMap(usedCCs, region.amplitudeEG.ccStart);
    collectUsedCCsFromCCMap(usedCCs, region.amplitudeEG.ccSustain);

    collectUsedCCsFromCCMap(usedCCs, region.ampVeltrackCC);
    collectUsedCCsFromCCMap(usedCCs, region.pitchVeltrackCC);
    for (const auto& filter: region.filters)
        collectUsedCCsFromCCMap(usedCCs, filter.veltrackCC);

    if (region.pitchEG) {
        collectUsedCCsFromCCMap(usedCCs, region.pitchEG->ccAttack);
        collectUsedCCsFromCCMap(usedCCs, region.pitchEG->ccRelease);
        collectUsedCCsFromCCMap(usedCCs, region.pitchEG->ccDecay);
        collectUsedCCsFromCCMap(usedCCs, region.pitchEG->ccDelay);
        collectUsedCCsFromCCMap(usedCCs, region.pitchEG->ccHold);
        collectUsedCCsFromCCMap(usedCCs, region.pitchEG->ccStart);
        collectUsedCCsFromCCMap(usedCCs, region.pitchEG->ccSustain);
    }

    if (region.filterEG) {
        collectUsedCCsFromCCMap(usedCCs, region.filterEG->ccAttack);
        collectUsedCCsFromCCMap(usedCCs, region.filterEG->ccRelease);
        collectUsedCCsFromCCMap(usedCCs, region.filterEG->ccDecay);
        collectUsedCCsFromCCMap(usedCCs, region.filterEG->ccDelay);
        collectUsedCCsFromCCMap(usedCCs, region.filterEG->ccHold);
        collectUsedCCsFromCCMap(usedCCs, region.filterEG->ccStart);
        collectUsedCCsFromCCMap(usedCCs, region.filterEG->ccSustain);
    }

    for (const LFODescription& lfo : region.lfos) {
        collectUsedCCsFromCCMap(usedCCs, lfo.phaseCC);
        collectUsedCCsFromCCMap(usedCCs, lfo.delayCC);
        collectUsedCCsFromCCMap(usedCCs, lfo.fadeCC);
    }
    for (const FlexEGDescription& flexEG : region.flexEGs) {
        for (const FlexEGPoint& point : flexEG.points) {
            collectUsedCCsFromCCMap(usedCCs, point.ccTime);
            collectUsedCCsFromCCMap(usedCCs, point.ccLevel);
        }
    }
    collectUsedCCsFromCCMap(usedCCs, region.ccConditions);
    collectUsedCCsFromCCMap(usedCCs, region.ccTriggers);
    collectUsedCCsFromCCMap(usedCCs, region.crossfadeCCInRange);
    collectUsedCCsFromCCMap(usedCCs, region.crossfadeCCOutRange);
}

void Synth::Impl::collectUsedCCsFromModulations(BitArray<config::numCCs>& usedCCs, const ModMatrix& mm)
{
    class CCSourceCollector : public ModMatrix::KeyVisitor {
    public:
        explicit CCSourceCollector(BitArray<config::numCCs>& used)
            : used_(used)
        {
        }

        bool visit(const ModKey& key) override
        {
            if (key.id() == ModId::Controller)
                used_.set(key.parameters().cc);
            return true;
        }
        BitArray<config::numCCs>& used_;
    };

    CCSourceCollector vtor(usedCCs);
    mm.visitSources(vtor);
}

BitArray<config::numCCs> Synth::Impl::collectAllUsedCCs()
{
    BitArray<config::numCCs> used;
    for (const LayerPtr& layerPtr : layers_) {
        collectUsedCCsFromRegion(used, layerPtr->getRegion());
        sustainOrSostenuto_.set(layerPtr->region_.sustainCC);
        sustainOrSostenuto_.set(layerPtr->region_.sostenutoCC);
    }
    collectUsedCCsFromModulations(used, resources_.getModMatrix());
    return used;
}

const std::string* Synth::Impl::getKeyLabel(int keyNumber) const
{
    auto it = keyLabelsMap_.find(keyNumber);
    return (it == keyLabelsMap_.end()) ? nullptr : &keyLabels_[it->second].second;
}

void Synth::Impl::setKeyLabel(int keyNumber, std::string name)
{
    auto it = keyLabelsMap_.find(keyNumber);
    if (it != keyLabelsMap_.end())
        keyLabels_[it->second].second = std::move(name);
    else {
        size_t index = keyLabels_.size();
        keyLabels_.emplace_back(keyNumber, std::move(name));
        keyLabelsMap_[keyNumber] = index;
    }
}

const std::string* Synth::Impl::getCCLabel(int ccNumber) const
{
    auto it = ccLabelsMap_.find(ccNumber);
    return (it == ccLabelsMap_.end()) ? nullptr : &ccLabels_[it->second].second;
}

void Synth::Impl::setCCLabel(int ccNumber, std::string name)
{
    auto it = ccLabelsMap_.find(ccNumber);
    if (it != ccLabelsMap_.end())
        ccLabels_[it->second].second = std::move(name);
    else {
        size_t index = ccLabels_.size();
        ccLabels_.emplace_back(ccNumber, std::move(name));
        ccLabelsMap_[ccNumber] = index;
    }
}

const std::string* Synth::Impl::getKeyswitchLabel(int swNumber) const
{
    auto it = keyswitchLabelsMap_.find(swNumber);
    return (it == keyswitchLabelsMap_.end()) ? nullptr : &keyswitchLabels_[it->second].second;
}

void Synth::Impl::setKeyswitchLabel(int swNumber, std::string name)
{
    auto it = keyswitchLabelsMap_.find(swNumber);
    if (it != keyswitchLabelsMap_.end())
        keyswitchLabels_[it->second].second = std::move(name);
    else {
        size_t index = keyswitchLabels_.size();
        keyswitchLabels_.emplace_back(swNumber, std::move(name));
        keyswitchLabelsMap_[swNumber] = index;
    }
}

void Synth::Impl::clearKeyLabels()
{
    keyLabels_.clear();
    keyLabelsMap_.clear();
}

void Synth::Impl::clearCCLabels()
{
    ccLabels_.clear();
    ccLabelsMap_.clear();
}

void Synth::Impl::clearKeyswitchLabels()
{
    keyswitchLabels_.clear();
    keyswitchLabelsMap_.clear();
}

Parser& Synth::getParser() noexcept
{
    Impl& impl = *impl_;
    return impl.parser_;
}

const Parser& Synth::getParser() const noexcept
{
    Impl& impl = *impl_;
    return impl.parser_;
}

const std::vector<NoteNamePair>& Synth::getKeyLabels() const noexcept
{
    Impl& impl = *impl_;
    return impl.keyLabels_;
}

const std::vector<CCNamePair>& Synth::getCCLabels() const noexcept
{
    Impl& impl = *impl_;
    return impl.ccLabels_;
}

Resources& Synth::getResources() noexcept
{
    Impl& impl = *impl_;
    return impl.resources_;
}

const Resources& Synth::getResources() const noexcept
{
    Impl& impl = *impl_;
    return impl.resources_;
}

} // namespace sfz
