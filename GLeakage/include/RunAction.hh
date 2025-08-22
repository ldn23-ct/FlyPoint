#ifndef RUNACTION_HH
#define RUNACTION_HH

#include "G4UserRunAction.hh"
#include "G4ThreeVector.hh"
#include "globals.hh"
#include <string>
#include <vector>
class G4Run;

class RunAction : public G4UserRunAction
{
  public:
    RunAction() {}
    ~RunAction() override = default;

    void DrawLine();
    static std::vector<G4ThreeVector> ReadPos(const std::string& path);
    void BeginOfRunAction(const G4Run*) override;
    void EndOfRunAction(const G4Run*) override;
    G4Run* GenerateRun() override;

};

#endif