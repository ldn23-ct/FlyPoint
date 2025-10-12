#include "SteppingAction.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Gamma.hh"
#include "G4Material.hh"
#include "G4VProcess.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4SystemOfUnits.hh"
#include "G4StepPoint.hh"

SteppingAction::SteppingAction(G4double dEthreshold, G4int verbose)
: fDEthr(dEthreshold),
  fVerbose(verbose),
  fMaxPrint(50),
  fPrinted(0)
{}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  // // 只关注 gamma
  // auto* trk = step->GetTrack();
  // if (trk->GetDefinition() != G4Gamma::Definition()) return;

  // auto* pre  = step->GetPreStepPoint();
  // auto* post = step->GetPostStepPoint();

  // // 能量（动能）
  // const G4double Epre  = pre->GetKineticEnergy();
  // const G4double Epost = post->GetKineticEnergy();
  // const G4double dE    = Epost - Epre;

  // // 1) 进入新体（几何边界）
  // if (post->GetStepStatus() == fGeomBoundary) {
  //   const auto* postVol = post->GetTouchableHandle().operator->()
  //                       ? post->GetTouchableHandle()->GetVolume()
  //                       : nullptr;
  //   const auto* preVol  = pre->GetTouchableHandle().operator->()
  //                       ? pre->GetTouchableHandle()->GetVolume()
  //                       : nullptr;
  //   // 进入体名称（可能为 nullptr：离开世界）
  //   const G4String postName = postVol ? postVol->GetName() : "NULL";
  //   const G4String preName  = preVol  ? preVol->GetName()  : "NULL";
  //   G4cout
  //     << "[ENTER] "
  //     << "TrackID=" << trk->GetTrackID()
  //     << "  " << preName << " -> " << postName
  //     << "  Ek=" << Epost/keV << "keV"
  //     << G4endl;
  // }

  // // 2) 动能变化超过 10 eV
  // if (std::fabs(dE) > 10.0*eV) {
  //   const G4VProcess* proc = post->GetProcessDefinedStep();
  //   const G4String pname = proc ? proc->GetProcessName() : "Unknown";
  //   const auto* preVol  = pre->GetTouchableHandle().operator->()
  //                       ? pre->GetTouchableHandle()->GetVolume()
  //                       : nullptr;
  //   const G4String preName  = preVol  ? preVol->GetName()  : "NULL";
  //   G4cout.setf(std::ios::fixed);
  //   G4cout << std::setprecision(3)
  //     << "[dE] "
  //     << "TrackID=" << trk->GetTrackID()
  //     << "  Ek(pre)="  << Epre/keV << "keV"
  //     << "  Ek(post)=" << Epost/keV << "keV"
  //     << "  dE="       << dE
  //     << "  proc="     << pname
  //     << "  Vol="      << preName
  //     << G4endl;
  // }
}
