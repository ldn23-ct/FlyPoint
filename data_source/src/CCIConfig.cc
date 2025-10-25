#include "CCIConfig.hh"
#include "CCIConfigMessenger.hh"
#include "G4SystemOfUnits.hh"

G4ThreadLocal CCIConfig* CCIConfig::fInstance = nullptr;

CCIConfig* CCIConfig::GetInstance()
{
  if (fInstance == nullptr) {
    fInstance = new CCIConfig();
  }
  return fInstance;
}

CCIConfig::CCIConfig()
{
  // Default Detector Parameters
  fDetNX = 50;
  fDetNY = 50;
  fDetSize = 1.0 * mm;
  fDetThick = 5.0 * mm;
  fDetPitch = 1.0 * mm;
  fParticleEnergy = 160.0 * keV;

  // Default Object dimensions
  fObjectX = 200.0 * mm;
  fObjectY = 200.0 * mm;
  fObjectZ = 70.0 * mm;

  // Default Detector position and rotation
  fDetCenter = G4ThreeVector(-76.175 * mm, 0.0, 119.96 * mm);
  fDetTheta = -40.0 * deg;

  // Default Slit parameters
  fSlit2DetDist = 20.0 * mm;
  fSlitThick = 4.0 * mm;
  fSlitWidth = 10.0 * mm;
  fSlitHeight = 1.0 * mm;

  // Default Primary Generator ParametersfXStart =-9.0 * mm;
  fXStart = -0.0 * mm;
  fXEnd = 0.0 * mm;
  fZPos = 243.0 * mm;
  fOmegaY = CLHEP::pi * 0.1 / 180.0; // 0.1 degree
  fVx = 3.0 * mm;
  fThetaMin = 0.0 * deg;
  fThetaMax= 0.0 * deg;

  fMessenger = new CCIConfigMessenger(this);
}

CCIConfig::~CCIConfig()
{
  delete fMessenger;
}
