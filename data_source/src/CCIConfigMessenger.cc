#include "CCIConfigMessenger.hh"
#include "CCIConfig.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4SystemOfUnits.hh"

CCIConfigMessenger::CCIConfigMessenger(CCIConfig* config)
  : G4UImessenger(), fConfig(config)
{
  fDir = new G4UIdirectory("/cci/config/");
  fDir->SetGuidance("UI commands to configure the simulation");

  // --- Existing Detector Commands ---
  fDetNXCmd = new G4UIcmdWithAnInteger("/cci/config/setDetNX", this);
  fDetNXCmd->SetGuidance("Set number of detector elements in X.");
  fDetNXCmd->SetParameterName("nx", false);
  fDetNXCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fDetNYCmd = new G4UIcmdWithAnInteger("/cci/config/setDetNY", this);
  fDetNYCmd->SetGuidance("Set number of detector elements in Y.");
  fDetNYCmd->SetParameterName("ny", false);
  fDetNYCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fDetSizeCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/setDetSize", this);
  fDetSizeCmd->SetGuidance("Set size of a single detector element.");
  fDetSizeCmd->SetParameterName("size", false);
  fDetSizeCmd->SetUnitCategory("Length");
  fDetSizeCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  fDetThickCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/setDetThick", this);
  fDetThickCmd->SetGuidance("Set thickness of a detector element.");
  fDetThickCmd->SetParameterName("thick", false);
  fDetThickCmd->SetUnitCategory("Length");
  fDetThickCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

  // --- Object Commands ---
  G4UIdirectory* objDir = new G4UIdirectory("/cci/config/object/");
  objDir->SetGuidance("Object dimension commands");

  fObjectXCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/object/setX", this);
  fObjectXCmd->SetGuidance("Set the X dimension of the object.");
  fObjectXCmd->SetParameterName("ObjectX", false);
  fObjectXCmd->SetUnitCategory("Length");
  fObjectXCmd->SetDefaultUnit("mm");

  fObjectYCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/object/setY", this);
  fObjectYCmd->SetGuidance("Set the Y dimension of the object.");
  fObjectYCmd->SetParameterName("ObjectY", false);
  fObjectYCmd->SetUnitCategory("Length");
  fObjectYCmd->SetDefaultUnit("mm");

  fObjectZCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/object/setZ", this);
  fObjectZCmd->SetGuidance("Set the Z dimension of the object.");
  fObjectZCmd->SetParameterName("ObjectZ", false);
  fObjectZCmd->SetUnitCategory("Length");
  fObjectZCmd->SetDefaultUnit("mm");

  // --- Detector Position/Rotation Commands ---
  G4UIdirectory* detPosDir = new G4UIdirectory("/cci/config/detector/");
  detPosDir->SetGuidance("Detector position and rotation commands");

  fDetCenterCmd = new G4UIcmdWith3VectorAndUnit("/cci/config/detector/setCenter", this);
  fDetCenterCmd->SetGuidance("Set the center position of the detector array.");
  fDetCenterCmd->SetParameterName("DetCenterX", "DetCenterY", "DetCenterZ", false);
  fDetCenterCmd->SetUnitCategory("Length");
  fDetCenterCmd->SetDefaultUnit("mm");

  fDetThetaCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/detector/setTheta", this);
  fDetThetaCmd->SetGuidance("Set the rotation angle of the detector array.");
  fDetThetaCmd->SetParameterName("DetTheta", false);
  fDetThetaCmd->SetUnitCategory("Angle");
  fDetThetaCmd->SetDefaultUnit("deg");

  // --- Slit Commands ---
  G4UIdirectory* slitDir = new G4UIdirectory("/cci/config/slit/");
  slitDir->SetGuidance("Slit dimension commands");

  fSlit2DetDistCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/slit/setDistanceToDetector", this);
  fSlit2DetDistCmd->SetGuidance("Set the distance from slit to detector.");
  fSlit2DetDistCmd->SetParameterName("Slit2DetDist", false);
  fSlit2DetDistCmd->SetUnitCategory("Length");
  fSlit2DetDistCmd->SetDefaultUnit("mm");

  fSlitThickCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/slit/setThickness", this);
  fSlitThickCmd->SetGuidance("Set the thickness of the slit.");
  fSlitThickCmd->SetParameterName("SlitThick", false);
  fSlitThickCmd->SetUnitCategory("Length");
  fSlitThickCmd->SetDefaultUnit("mm");

  fSlitWidthCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/slit/setWidth", this);
  fSlitWidthCmd->SetGuidance("Set the width of the slit opening.");
  fSlitWidthCmd->SetParameterName("SlitWidth", false);
  fSlitWidthCmd->SetUnitCategory("Length");
  fSlitWidthCmd->SetDefaultUnit("mm");

  fSlitHeightCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/slit/setHeight", this);
  fSlitHeightCmd->SetGuidance("Set the height of the slit opening.");
  fSlitHeightCmd->SetParameterName("SlitHeight", false);
  fSlitHeightCmd->SetUnitCategory("Length");
  fSlitHeightCmd->SetDefaultUnit("mm");

  // --- Source Commands ---
  G4UIdirectory* sourceDir = new G4UIdirectory("/cci/config/source/");
  sourceDir->SetGuidance("Primary particle generator commands");

  fXStartCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/source/setXStart", this);
  fXStartCmd->SetGuidance("Set the starting X position of the source scan.");
  fXStartCmd->SetParameterName("XStart", false);
  fXStartCmd->SetUnitCategory("Length");
  fXStartCmd->SetDefaultUnit("mm");

  fXEndCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/source/setXEnd", this);
  fXEndCmd->SetGuidance("Set the ending X position of the source scan.");
  fXEndCmd->SetParameterName("XEnd", false);
  fXEndCmd->SetUnitCategory("Length");
  fXEndCmd->SetDefaultUnit("mm");

  fZPosCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/source/setZPos", this);
  fZPosCmd->SetGuidance("Set the Z position of the source.");
  fZPosCmd->SetParameterName("ZPos", false);
  fZPosCmd->SetUnitCategory("Length");
  fZPosCmd->SetDefaultUnit("mm");

  fOmegaYCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/source/setOmegaY", this);
  fOmegaYCmd->SetGuidance("Set the angular velocity of the source scan.");
  fOmegaYCmd->SetParameterName("OmegaY", false);
  fOmegaYCmd->SetUnitCategory("Angle");
  fOmegaYCmd->SetDefaultUnit("rad");

  fVxCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/source/setVx", this);
  fVxCmd->SetGuidance("Set the linear velocity of the source scan.");
  fVxCmd->SetParameterName("Vx", false);
  fVxCmd->SetUnitCategory("Length");
  fVxCmd->SetDefaultUnit("mm");

  fThetaMinCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/source/setThetaMin", this);
  fThetaMinCmd->SetGuidance("Set the minimum emission angle.");
  fThetaMinCmd->SetParameterName("ThetaMin", false);
  fThetaMinCmd->SetUnitCategory("Angle");
  fThetaMinCmd->SetDefaultUnit("rad");

  fThetaMaxCmd = new G4UIcmdWithADoubleAndUnit("/cci/config/source/setThetaMax", this);
  fThetaMaxCmd->SetGuidance("Set the maximum emission angle.");
  fThetaMaxCmd->SetParameterName("ThetaMax", false);
  fThetaMaxCmd->SetUnitCategory("Angle");
  fThetaMaxCmd->SetDefaultUnit("rad");
}

CCIConfigMessenger::~CCIConfigMessenger()
{
  delete fDetNXCmd;
  delete fDetNYCmd;
  delete fDetSizeCmd;
  delete fDetThickCmd;
  delete fDetCenterCmd;
  delete fDetThetaCmd;
  delete fParticleEnergyCmd;
  delete fObjectXCmd;
  delete fObjectYCmd;
  delete fObjectZCmd;
  delete fDetCenterCmd;
  delete fDetThetaCmd;
  delete fSlit2DetDistCmd;
  delete fSlitThickCmd;
  delete fSlitWidthCmd;
  delete fSlitHeightCmd;
  delete fXStartCmd;
  delete fXEndCmd;
  delete fZPosCmd;
  delete fOmegaYCmd;
  delete fVxCmd;
  delete fThetaMinCmd;
  delete fThetaMaxCmd;
  delete fDir;
}

void CCIConfigMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == fDetNXCmd) {
        fConfig->SetDetNX(fDetNXCmd->GetNewIntValue(newValue));
    } else if (command == fDetNYCmd) {
        fConfig->SetDetNY(fDetNYCmd->GetNewIntValue(newValue));
    } else if (command == fDetSizeCmd) {
        fConfig->SetDetSize(fDetSizeCmd->GetNewDoubleValue(newValue));
    } else if (command == fDetThickCmd) {
        fConfig->SetDetThick(fDetThickCmd->GetNewDoubleValue(newValue));
    } else if (command == fObjectXCmd) {
    fConfig->SetObjectX(fObjectXCmd->GetNewDoubleValue(newValue));
  } else if (command == fObjectYCmd) {
    fConfig->SetObjectY(fObjectYCmd->GetNewDoubleValue(newValue));
  } else if (command == fObjectZCmd) {
    fConfig->SetObjectZ(fObjectZCmd->GetNewDoubleValue(newValue));
  } else if (command == fDetCenterCmd) {
        fConfig->SetDetCenter(fDetCenterCmd->GetNew3VectorValue(newValue));
    } else if (command == fDetThetaCmd) {
        fConfig->SetDetTheta(fDetThetaCmd->GetNewDoubleValue(newValue));
    } else if (command == fSlit2DetDistCmd) {
        fConfig->SetSlit2DetDist(fSlit2DetDistCmd->GetNewDoubleValue(newValue));
    } else if (command == fSlitThickCmd) {
        fConfig->SetSlitThick(fSlitThickCmd->GetNewDoubleValue(newValue));
    } else if (command == fSlitWidthCmd) {
        fConfig->SetSlitWidth(fSlitWidthCmd->GetNewDoubleValue(newValue));
    } else if (command == fSlitHeightCmd) {
        fConfig->SetSlitHeight(fSlitHeightCmd->GetNewDoubleValue(newValue));
    } else if (command == fXStartCmd) {
    fConfig->SetXStart(fXStartCmd->GetNewDoubleValue(newValue));
  } else if (command == fXEndCmd) {
    fConfig->SetXEnd(fXEndCmd->GetNewDoubleValue(newValue));
  } else if (command == fZPosCmd) {
    fConfig->SetZPos(fZPosCmd->GetNewDoubleValue(newValue));
  } else if (command == fOmegaYCmd) {
    fConfig->SetOmegaY(fOmegaYCmd->GetNewDoubleValue(newValue));
  } else if (command == fVxCmd) {
    fConfig->SetVx(fVxCmd->GetNewDoubleValue(newValue));
  } else if (command == fThetaMinCmd) {
    fConfig->SetThetaMin(fThetaMinCmd->GetNewDoubleValue(newValue));
  } else if (command == fThetaMaxCmd) {
    fConfig->SetThetaMax(fThetaMaxCmd->GetNewDoubleValue(newValue));
  }
}
