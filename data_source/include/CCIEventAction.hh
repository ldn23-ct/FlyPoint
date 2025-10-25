#ifndef CCIEventAction_h
#define CCIEventAction_h

#include "G4UserEventAction.hh"
#include "G4Types.hh"
#include "G4ThreeVector.hh" // 用于位置和方向
#include <map>
#include <vector>

class G4Event;

// --- 新增：用于存储单次散射事件信息的结构体 ---
struct ScatterEventInfo {
    G4ThreeVector position;    // 散射发生的位置
    G4ThreeVector direction;   // 散射后光子的方向
    bool isValid = false;      // 标记此信息是否有效
};

class CCIEventAction : public G4UserEventAction {
public:
    CCIEventAction();
    ~CCIEventAction() override;
    void BeginOfEventAction(const G4Event*) override;
    void EndOfEventAction(const G4Event*) override;

    void SetPrimaryTheta(G4double theta) { fPrimaryTheta = theta; }
    G4double GetPrimaryTheta() const { return fPrimaryTheta; }    

    // --- 现有方法保持不变 ---
    static void AddScatterCount(G4int trackID);
    static G4int GetScatterCount(G4int trackID);
    static void SetEntryScatterCount(G4int trackID, G4int count);
    static G4int GetEntryScatterCount(G4int trackID);
    static void SetParent(G4int trackID, G4int parentID);
    static G4int GetAncestorScatterCount(G4int trackID);
    static G4int GetAncestorEntryScatterCount(G4int trackID);

    // --- 用于记录所有散射事件的方法 ---
    static void RecordScatterEvent(G4int trackID, const G4ThreeVector& pos, const G4ThreeVector& dir);
    static std::vector<ScatterEventInfo> GetAncestorScatterEvents(G4int trackID);

private:
    std::map<G4int, G4int> fScatterCounts;
    std::map<G4int, G4int> fParentMap;
    std::map<G4int, G4int> fEntryScatterCounts;

    G4double fPrimaryTheta;

    // --- 新增：存储所有散射事件的 map ---
    std::map<G4int, std::vector<ScatterEventInfo>> fAllScatterInfo;

    // 线程本地实例指针，供静态接口使用
    static G4ThreadLocal CCIEventAction* fgInstance;
};

#endif