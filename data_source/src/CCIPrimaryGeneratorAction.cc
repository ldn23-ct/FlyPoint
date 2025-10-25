#include "CCIPrimaryGeneratorAction.hh"
#include "CCIConfig.hh"
#include "CCIEventAction.hh"

#include "G4RunManager.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4Geantino.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

CCIPrimaryGeneratorAction::CCIPrimaryGeneratorAction()
 : G4VUserPrimaryGeneratorAction(),
   fParticleGun(0)
{
  G4int nofParticles = 1;
  fParticleGun = new G4ParticleGun(nofParticles);
  // default particle kinematic
  //
  G4ParticleDefinition* particleDefinition
    = G4ParticleTable::GetParticleTable()->FindParticle("gamma");
  fParticleGun->SetParticleDefinition(particleDefinition);
  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0,0,-1));
  fParticleGun->SetParticleEnergy(CCIConfig::GetInstance()->GetParticleEnergy());
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

CCIPrimaryGeneratorAction::~CCIPrimaryGeneratorAction()
{
  delete fParticleGun;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void CCIPrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
  // Get config instance
  auto config = CCIConfig::GetInstance();

  // 参数
  G4double x_start = config->GetXStart();
  G4double x_end = config->GetXEnd();
  G4double z_pos = config->GetZPos();
  G4double omega_y = config->GetOmegaY();
  G4double vx = config->GetVx();
  G4int nEvents = G4RunManager::GetRunManager()->GetNumberOfEventsToBeProcessed();

  // y方向角度采样范围
  G4double theta_min = config->GetThetaMin();
  G4double theta_max = config->GetThetaMax();
  G4double theta_range = theta_max - theta_min;

  // 计算每个x点采样多少个角度
  G4int n_theta = (G4int)(theta_range / omega_y) + 1;
  G4double dtheta = 0.0;
  if (n_theta > 1) {
    dtheta = theta_range / (n_theta - 1);
  } else {
    // 只有一个角度时，步长为0
    dtheta = 0.0;
  }

  // 计算x方向步数
  G4int n_x = (G4int)((x_end - x_start) / vx) + 1;

  // 计算每个(x,theta)组合的采样数
  G4int n_per = nEvents / (n_x * n_theta);

  // 当前eventID
  G4int eventID = anEvent->GetEventID();

  // 计算当前x、theta、采样索引
  G4int i_combo = eventID / n_per;
  G4int isample = eventID % n_per;
  G4int ix = i_combo / n_theta;
  G4int itheta = i_combo % n_theta;

  // 超出范围则不发射
  if(ix >= n_x) return;

  // 计算当前x和theta
  G4double x = x_start + ix * vx;
  G4double theta = theta_max - itheta * dtheta;

  auto eventAction = const_cast<CCIEventAction*>(
      static_cast<const CCIEventAction*>(
          G4RunManager::GetRunManager()->GetUserEventAction()
      )
  );
  if (eventAction) {
      eventAction->SetPrimaryTheta(theta);
  }

  // 粒子初始位置
  G4ThreeVector pos(x, 0, z_pos);

  // 粒子动量方向
  G4ThreeVector dir(0, sin(theta), -cos(theta));
  dir = dir.unit();

  fParticleGun->SetParticlePosition(pos);
  fParticleGun->SetParticleMomentumDirection(dir);
  fParticleGun->GeneratePrimaryVertex(anEvent);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
