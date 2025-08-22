#ifndef AIRSD_HH
#define AIRSD_HH

#include "G4VSensitiveDetector.hh"
#include <unordered_set>

class G4Step;
class G4HCofThisEvent;
class Run;

class AirSD: public G4VSensitiveDetector
{
public:
    AirSD(G4String name);
    ~AirSD() override = default;

    void Initialize(G4HCofThisEvent*) override;
    G4bool ProcessHits(G4Step* aStep, G4TouchableHistory*) override;

private:
    std::unordered_set<G4int> m_seenTrackIDs;
    Run* run = nullptr;
    G4int ncos;
    G4int nphi;
    G4int ne;
    G4double e_start;
    G4double e_width;
    G4double r;
};

#endif