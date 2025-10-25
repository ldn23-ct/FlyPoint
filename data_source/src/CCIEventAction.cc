#include "CCIEventAction.hh"
#include "G4Event.hh"
#include "CCILayerSD.hh"
#include "G4SDManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4HCofThisEvent.hh"
#include "G4THitsCollection.hh"

#include <fstream>
#include <algorithm> // 用于 std::sort 等（如果需要排序）

#include "G4AutoLock.hh"
#include "G4Threading.hh"

namespace { G4Mutex spectrumFileMutex = G4MUTEX_INITIALIZER; }

// 线程本地实例初始化
G4ThreadLocal CCIEventAction* CCIEventAction::fgInstance = nullptr;

CCIEventAction::CCIEventAction(): fPrimaryTheta(0.0)
{
    fgInstance = this;
}

CCIEventAction::~CCIEventAction()
{
    if (fgInstance == this) fgInstance = nullptr;
}

void CCIEventAction::BeginOfEventAction(const G4Event*)
{
    // 清空每事件的数据
    fScatterCounts.clear();
    fParentMap.clear();
    fEntryScatterCounts.clear();
    fAllScatterInfo.clear(); // 清空所有散射事件的记录
}

void CCIEventAction::EndOfEventAction(const G4Event* event)
{
    if (!event) return;
    G4SDManager* sdMan = G4SDManager::GetSDMpointer();
    G4int hcID = sdMan->GetCollectionID("LayerSD/LayerHits");
    if (hcID < 0) return;
    G4HCofThisEvent* hce = event->GetHCofThisEvent();
    if (!hce) return;
    auto hitsCol = static_cast<G4THitsCollection<CCILayerHit>*>(hce->GetHC(hcID));
    if (!hitsCol || hitsCol->entries() == 0) return;

    // --- 找到时间最早的那个 Hit ---
    CCILayerHit* firstHit = nullptr;
    for (G4int i = 0; i < hitsCol->entries(); ++i) {
        CCILayerHit* currentHit = (*hitsCol)[i];
        if (!currentHit) continue;
        if (!firstHit || currentHit->time < firstHit->time) {
            firstHit = currentHit;
        }
    }

    // --- 如果找到了最早的 hit，则处理并输出 ---
    if (firstHit) {
        G4int trackID = firstHit->trackID;
        
        // 获取所有散射事件
        std::vector<ScatterEventInfo> allScatters = GetAncestorScatterEvents(trackID);

        // 只处理那些确实发生了散射的事件
        if (!allScatters.empty()) {
            G4AutoLock lock(&spectrumFileMutex);
            std::ofstream ofs("../output/spectrum_0_interval_case1.txt", std::ios::app);
            if (!ofs.is_open()) {
                G4cerr << "ERROR: Failed to open output/spectrum.txt for writing." << G4endl;
                return;
            }

            G4int evtID = event->GetEventID();
            G4int copyNo  = firstHit->copyNo;
            G4double kinE = firstHit->kinE;
            G4int sc = GetAncestorEntryScatterCount(trackID);

            ofs << evtID << " "
                << copyNo << " "
                << kinE / keV << " "
                << fPrimaryTheta / deg << " "
                << sc; // sc 应该等于 allScatters.size()

            // 循环遍历并输出每一次散射的信息
            // for (const auto& scatter : allScatters) {
            //     ofs << " "
            //         << scatter.position.x() / mm << " "
            //         << scatter.position.y() / mm << " "
            //         << scatter.position.z() / mm << " "
            //         << scatter.direction.x() << " "
            //         << scatter.direction.y() << " "
            //         << scatter.direction.z();
            // }
            ofs << "\n";
        }
    }
}

// --- 现有函数的实现 (保持不变) ---

void CCIEventAction::AddScatterCount(G4int trackID)
{
    if (!fgInstance) return;
    fgInstance->fScatterCounts[trackID] += 1;
}

G4int CCIEventAction::GetScatterCount(G4int trackID)
{
    if (!fgInstance) return 0;
    auto it = fgInstance->fScatterCounts.find(trackID);
    return (it != fgInstance->fScatterCounts.end()) ? it->second : 0;
}

void CCIEventAction::SetParent(G4int trackID, G4int parentID)
{
    if (!fgInstance) return;
    if (fgInstance->fParentMap.find(trackID) == fgInstance->fParentMap.end()) {
        fgInstance->fParentMap[trackID] = parentID;
    }
}

G4int CCIEventAction::GetAncestorScatterCount(G4int trackID)
{
    if (!fgInstance) return 0;
    G4int total = 0;
    G4int depth = 0;
    const G4int maxDepth = 1024;
    G4int cur = trackID;
    while (cur > 0 && depth < maxDepth) {
        total += GetScatterCount(cur);
        auto pit = fgInstance->fParentMap.find(cur);
        if (pit == fgInstance->fParentMap.end()) break;
        cur = pit->second;
        ++depth;
    }
    return total;
}

void CCIEventAction::SetEntryScatterCount(G4int trackID, G4int count)
{
    if (!fgInstance) return;
    if (fgInstance->fEntryScatterCounts.find(trackID) == fgInstance->fEntryScatterCounts.end()) {
        fgInstance->fEntryScatterCounts[trackID] = count;
    }
}

G4int CCIEventAction::GetEntryScatterCount(G4int trackID)
{
    if (!fgInstance) return 0;
    auto it = fgInstance->fEntryScatterCounts.find(trackID);
    return (it != fgInstance->fEntryScatterCounts.end()) ? it->second : 0;
}

G4int CCIEventAction::GetAncestorEntryScatterCount(G4int trackID)
{
    if (!fgInstance) return 0;
    G4int depth = 0;
    const G4int maxDepth = 1024;
    G4int cur = trackID;
    while (cur > 0 && depth < maxDepth) {
        auto it = fgInstance->fEntryScatterCounts.find(cur);
        if (it != fgInstance->fEntryScatterCounts.end()) {
            return it->second;
        }
        auto pit = fgInstance->fParentMap.find(cur);
        if (pit == fgInstance->fParentMap.end()) break;
        cur = pit->second;
        ++depth;
    }
    return 0;
}

// --- 新增：记录和获取所有散射事件的函数实现 ---

void CCIEventAction::RecordScatterEvent(G4int trackID, const G4ThreeVector& pos, const G4ThreeVector& dir)
{
    if (!fgInstance) return;
    ScatterEventInfo newEvent;
    newEvent.position = pos;
    newEvent.direction = dir;
    newEvent.isValid = true;
    fgInstance->fAllScatterInfo[trackID].push_back(newEvent);
}

std::vector<ScatterEventInfo> CCIEventAction::GetAncestorScatterEvents(G4int trackID)
{
    std::vector<ScatterEventInfo> allEvents;
    if (!fgInstance) return allEvents;

    G4int depth = 0;
    const G4int maxDepth = 1024;
    G4int cur = trackID;
    while (cur > 0 && depth < maxDepth) {
        auto it = fgInstance->fAllScatterInfo.find(cur);
        if (it != fgInstance->fAllScatterInfo.end()) {
            allEvents.insert(allEvents.end(), it->second.begin(), it->second.end());
        }
        auto pit = fgInstance->fParentMap.find(cur);
        if (pit == fgInstance->fParentMap.end()) break;
        cur = pit->second;
        ++depth;
    }
    return allEvents;
}