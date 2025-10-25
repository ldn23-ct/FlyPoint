#include "CCILayerSD.hh"
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4TouchableHistory.hh"
#include "G4Track.hh"
#include "G4ios.hh"

CCILayerSD::CCILayerSD(const G4String& name)
  : G4VSensitiveDetector(name), fHitsCollection(nullptr), fHitsCollectionID(-1)
{
  collectionName.insert("LayerHits");
}

CCILayerSD::~CCILayerSD() {}

void CCILayerSD::Initialize(G4HCofThisEvent* hce)
{
  fHitsCollection = new G4THitsCollection<CCILayerHit>(SensitiveDetectorName, collectionName[0]);
  if (fHitsCollectionID < 0) {
    fHitsCollectionID = G4SDManager::GetSDMpointer()->GetCollectionID(fHitsCollection);
  }
  hce->AddHitsCollection(fHitsCollectionID, fHitsCollection);
}

G4bool CCILayerSD::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
  if (!aStep) return false;
  G4StepPoint* pre = aStep->GetPreStepPoint();
  if (!pre) return false;
  G4Track* track = aStep->GetTrack();
  if (!track) return false;

  // 不以 edep 过滤，只记录进入探测器时的粒子信息（或所有 step）
  G4int tid = track->GetTrackID();
  G4int pid = track->GetParentID();
  G4int copyNo = 0;
  if (pre->GetTouchable()) copyNo = pre->GetTouchable()->GetCopyNumber();
  G4double kinE = pre->GetKineticEnergy(); // MeV
  G4double time = pre->GetGlobalTime();    // ns
  G4ThreeVector pos = pre->GetPosition();
  G4String pname = track->GetDefinition()->GetParticleName();

  CCILayerHit* hit = new CCILayerHit(tid, copyNo, pid, kinE, time, pos, pname);
  if (fHitsCollection) fHitsCollection->insert(hit);

  return true;
}

void CCILayerSD::EndOfEvent(G4HCofThisEvent* ) {}

void CCILayerHit::Print() const
{
  G4cout << "CCILayerHit: track=" << trackID
         << " copy=" << copyNo
         << " kinE=" << kinE/keV << " keV"
         << " pos=" << pos << G4endl;
}