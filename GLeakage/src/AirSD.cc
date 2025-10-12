#include "AirSD.hh"
#include "Config.hh"
#include "Run.hh"

#include "G4RunManager.hh"
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4VTouchable.hh"
#include "G4Gamma.hh"
#include "G4Threading.hh"
#include <stdexcept>
// #include <atomic>
// static std::atomic<int> printed{0};


AirSD::AirSD(G4String name)
  : G4VSensitiveDetector(name)
{}

void AirSD::Initialize(G4HCofThisEvent* hitsCE)
{
    const Config& config = Config::Instance();
    ncos = config.GetNcos();
    nphi = config.GetNphi();
    ne = config.GetNe();
    e_start = config.GetEstart();
    e_width = config.GetEwidth();
    r = config.GetR1();

    run = static_cast<Run*>(
      G4RunManager::GetRunManager()->GetNonConstCurrentRun());
    m_seenTrackIDs.clear();

}

G4bool AirSD::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
    G4Track* track = aStep->GetTrack();
    // auto post = aStep->GetPostStepPoint();
    auto pre = aStep->GetPreStepPoint();
    G4int id = track->GetTrackID();
    if(track->GetDefinition() != G4Gamma::Definition()) return false;
    if(pre->GetStepStatus() != fGeomBoundary) return false;
    if (!m_seenTrackIDs.insert(id).second) {
        return false; // 这个 track 已经记过了
    }

    G4ThreeVector pos = pre->GetPosition();
    G4double e = pre->GetKineticEnergy();

    // bin
    G4double costheta = pos.z()/r;
    costheta = std::clamp(costheta, -1.0, 1.0);
    G4double phi = pos.phi();
    if(phi < 0) phi += 2*M_PI;
    G4int icos = G4int((costheta + 1) / 2 * ncos);
    if(icos >= ncos) icos = ncos - 1;
    G4int iphi = G4int(phi / (2*M_PI) * nphi);
    if(iphi >= nphi) iphi = nphi - 1;
    G4int idir = icos*nphi + iphi;
    G4int ie = G4int( (e - e_start) / e_width);
    if(ie < 0) ie = 0;
    else if(ie >= ne) ie = ne - 1;
    
    run->AddCnts(idir, ie);
    track->SetTrackStatus(fStopAndKill);

    // // 限流打印：只打印前 50 条
    // if (printed.fetch_add(1) < 50) {
    //     G4cout << "[SD-Hit] dir=" << idir
    //             << "x=" << pos.x()
    //             << "y=" << pos.y()
    //             << "z=" << pos.z()
    //             << "cos=" << costheta
    //             << "phi=" << phi
    //             << "icos=" << icos
    //             << "iphi=" << iphi
    //             << " ie=" << ie
    //             << " Ek=" << e/CLHEP::keV << " keV"
    //             << G4endl;
    // }

    return true;
}
