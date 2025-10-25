#include "CCISteppingAction.hh"
#include "CCIEventAction.hh"
#include "CCIDetectorConstruction.hh"

#include "G4String.hh"

#include "G4Step.hh"
#include "G4RunManager.hh"
#include "G4ios.hh"
#include "G4Track.hh"

#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"

#include <iomanip>

#include <fstream>
using namespace std;
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

CCISteppingAction::CCISteppingAction(
                      const CCIDetectorConstruction* detectorConstruction,
                      CCIEventAction* eventAction)
  : G4UserSteppingAction(),
    fDetConstruction(detectorConstruction),
    fEventAction(eventAction)
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

CCISteppingAction::~CCISteppingAction()
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void CCISteppingAction::UserSteppingAction(const G4Step* step)
{
    if (!step) return;
    G4Track* track = step->GetTrack();
    if (!track) return;

    // 1. 记录 parent 映射（保持不变）
    G4int tid = track->GetTrackID();
    G4int pid = track->GetParentID();
    CCIEventAction::SetParent(tid, pid);

    // 获取步进点
    G4StepPoint* pre = step->GetPreStepPoint();
    G4StepPoint* post = step->GetPostStepPoint();
    if (!pre || !post) return;

    // 2. 新增：检测是否进入探测器并创建快照
    G4VPhysicalVolume* pvPre = pre->GetPhysicalVolume();
    G4VPhysicalVolume* pvPost = post->GetPhysicalVolume();
    // 条件：step 跨越了物理体边界，并且终点在探测器内
    if (pvPost && pvPost != pvPre) {
        G4String postName = pvPost->GetName();
        // 假设你的探测器名字包含 "Layer" 或 "Detector"
        if (postName.find("layer") != G4String::npos) {
            // 在此刻，获取该 track 沿祖先链的散射总数
            G4int sc_snapshot = CCIEventAction::GetAncestorScatterCount(tid);
            // 保存这个快照值
            CCIEventAction::SetEntryScatterCount(tid, sc_snapshot);
        }
    }

    // 3. 散射计数逻辑（保持不变，但只对光子和目标体）
    const G4ParticleDefinition* pdef = track->GetDefinition();
    if (!pdef || pdef->GetPDGEncoding() != 22) return; // 只关注光子

    const G4VProcess* proc = post->GetProcessDefinedStep();
    if (!proc) return;
    G4String pname = proc->GetProcessName();

    if (pname.find("compt") != G4String::npos || pname.find("Compton") != G4String::npos) {
        G4String volName = pvPost ? pvPost->GetName() : "";
        if (volName.find("object") != G4String::npos) {
            // 1. 累加散射次数（保持不变）
            CCIEventAction::AddScatterCount(tid);

            // 2. 修改：记录每一次散射事件
            G4ThreeVector scatterPos = post->GetPosition();
            G4ThreeVector scatterDir = post->GetMomentumDirection();
            CCIEventAction::RecordScatterEvent(tid, scatterPos, scatterDir); // 调用新函数
        }
    }
}






//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
