#include "CCIActionInitialization.hh"
#include "CCIPrimaryGeneratorAction.hh"
#include "CCIRunAction.hh"
#include "CCIEventAction.hh"
#include "CCISteppingAction.hh"
#include "CCIDetectorConstruction.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

CCIActionInitialization::CCIActionInitialization
                            (CCIDetectorConstruction* detConstruction)
 : G4VUserActionInitialization(),
   fDetConstruction(detConstruction)
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

CCIActionInitialization::~CCIActionInitialization()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void CCIActionInitialization::BuildForMaster() const
{
  SetUserAction(new CCIRunAction);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void CCIActionInitialization::Build() const
{
    // Worker / single-thread: 注册所有用户动作
    SetUserAction(new CCIPrimaryGeneratorAction());
    CCIEventAction* eventAction = new CCIEventAction();
    SetUserAction(eventAction);
    // 如果你的 SteppingAction 需要 eventAction 与 detector 指针
    SetUserAction(new CCISteppingAction(fDetConstruction, eventAction));
    // RunAction（可选）
    SetUserAction(new CCIRunAction());
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
