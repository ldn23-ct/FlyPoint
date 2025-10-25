#ifndef CCIRunAction_h
#define CCIRunAction_h 1

#include "G4UserRunAction.hh"
#include "globals.hh"

class G4Run;


class CCIRunAction : public G4UserRunAction
{
  public:
    CCIRunAction();
    virtual ~CCIRunAction();


    virtual void BeginOfRunAction(const G4Run*);
    virtual void   EndOfRunAction(const G4Run*);
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
