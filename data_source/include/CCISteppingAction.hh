#ifndef CCISteppingAction_h
#define CCISteppingAction_h 1

#include "G4UserSteppingAction.hh"

class CCIDetectorConstruction;
class CCIEventAction;

class CCISteppingAction : public G4UserSteppingAction
{
public:
  CCISteppingAction(const CCIDetectorConstruction* detectorConstruction,
                    CCIEventAction* eventAction);
  virtual ~CCISteppingAction();

  virtual void UserSteppingAction(const G4Step* step);

private:
  const CCIDetectorConstruction* fDetConstruction;
  CCIEventAction*  fEventAction;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
