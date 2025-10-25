#ifndef CCICONFIG_HH
#define CCICONFIG_HH

#include "globals.hh"
#include "G4ThreeVector.hh"

class CCIConfigMessenger; // Forward declaration

class CCIConfig
{
public:
  static CCIConfig* GetInstance();
  ~CCIConfig();

private:
  CCIConfig();
  static G4ThreadLocal CCIConfig* fInstance;
  CCIConfigMessenger* fMessenger;

public:
  // Detector Parameters
  G4int GetDetNX() const { return fDetNX; }
  void SetDetNX(G4int nx) { fDetNX = nx; }

  G4int GetDetNY() const { return fDetNY; }
  void SetDetNY(G4int ny) { fDetNY = ny; }

  G4double GetDetSize() const { return fDetSize; }
  void SetDetSize(G4double val) { fDetSize = val; }

  G4double GetDetThick() const { return fDetThick; }
  void SetDetThick(G4double val) { fDetThick = val; }

  G4double GetDetPitch() const { return fDetPitch; }
  void SetDetPitch(G4double val) { fDetPitch = val; }

  G4double GetParticleEnergy() const { return fParticleEnergy; }
  void SetParticleEnergy(G4double val) { fParticleEnergy = val; }

  // Object dimensions
  G4double GetObjectX() const { return fObjectX; }
  void SetObjectX(G4double val) { fObjectX = val; }
  G4double GetObjectY() const { return fObjectY; }
  void SetObjectY(G4double val) { fObjectY = val; }
  G4double GetObjectZ() const { return fObjectZ; }
  void SetObjectZ(G4double val) { fObjectZ = val; }

  // Detector position and rotation
  const G4ThreeVector& GetDetCenter() const { return fDetCenter; }
  void SetDetCenter(const G4ThreeVector& pos) { fDetCenter = pos; }

  G4double GetDetTheta() const { return fDetTheta; }
  void SetDetTheta(G4double val) { fDetTheta = val; }

  // Slit parameters
  G4double GetSlit2DetDist() const { return fSlit2DetDist; }
  void SetSlit2DetDist(G4double val) { fSlit2DetDist = val; }
  G4double GetSlitThick() const { return fSlitThick; }
  void SetSlitThick(G4double val) { fSlitThick = val; }
  G4double GetSlitWidth() const { return fSlitWidth; }
  void SetSlitWidth(G4double val) { fSlitWidth = val; }
  G4double GetSlitHeight() const { return fSlitHeight; }
  void SetSlitHeight(G4double val) { fSlitHeight = val; }

  // Primary Generator Parameters
  G4double GetXStart() const { return fXStart; }
  void SetXStart(G4double val) { fXStart = val; }
  G4double GetXEnd() const { return fXEnd; }
  void SetXEnd(G4double val) { fXEnd = val; }
  G4double GetZPos() const { return fZPos; }
  void SetZPos(G4double val) { fZPos = val; }
  G4double GetOmegaY() const { return fOmegaY; }
  void SetOmegaY(G4double val) { fOmegaY = val; }
  G4double GetVx() const { return fVx; }
  void SetVx(G4double val) { fVx = val; }
  G4double GetThetaMin() const { return fThetaMin; }
  void SetThetaMin(G4double val) { fThetaMin = val; }
  G4double GetThetaMax() const { return fThetaMax; }
  void SetThetaMax(G4double val) { fThetaMax = val; }

private:
  // Detector Parameters
  G4int fDetNX;
  G4int fDetNY;
  G4double fDetSize;
  G4double fDetThick;
  G4double fDetPitch;
  G4double fParticleEnergy;

  // New Parameters
  G4double fObjectX;
  G4double fObjectY;
  G4double fObjectZ;
  G4ThreeVector fDetCenter;
  G4double fDetTheta;
  G4double fSlit2DetDist;
  G4double fSlitThick;
  G4double fSlitWidth;
  G4double fSlitHeight;

  // Primary Generator Parameters
  G4double fXStart;
  G4double fXEnd;
  G4double fZPos;
  G4double fOmegaY;
  G4double fVx;
  G4double fThetaMin;
  G4double fThetaMax;
};

#endif
