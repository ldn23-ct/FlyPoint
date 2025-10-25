#ifndef CCILayerSD_h
#define CCILayerSD_h

#include "G4VSensitiveDetector.hh"
#include "G4THitsCollection.hh"
#include "G4VHit.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4String.hh"

class CCILayerHit : public G4VHit {
public:
  // 参数： trackID, copyNo, parentID, kinE(MeV), time(ns), position, particleName
  CCILayerHit(G4int tid=0, G4int cp=0, G4int pid=0,
              G4double ke=0.0, G4double t=0.0,
              const G4ThreeVector& p=G4ThreeVector(),
              const G4String& pn = "")
    : trackID(tid), copyNo(cp), parentID(pid), kinE(ke), time(t), pos(p), pName(pn) {}
  ~CCILayerHit() override {}

  G4int    trackID;
  G4int    copyNo;
  G4int    parentID;
  G4double kinE; // MeV (pre-step kinetic energy)
  G4double time; // ns (pre-step global time)
  G4ThreeVector pos;
  G4String pName; // particle name, e.g. "gamma"

  void Print() const;
};

class CCILayerSD : public G4VSensitiveDetector {
public:
  CCILayerSD(const G4String& name);
  ~CCILayerSD() override;

  void Initialize(G4HCofThisEvent* hce) override;
  G4bool ProcessHits(G4Step* aStep, G4TouchableHistory* ROhist) override;
  void EndOfEvent(G4HCofThisEvent* hce) override;

private:
  G4THitsCollection<CCILayerHit>* fHitsCollection{nullptr};
  G4int fHitsCollectionID{-1};
};

#endif