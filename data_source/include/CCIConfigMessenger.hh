#pragma once

#include "G4UImessenger.hh"
#include "globals.hh"

class CCIConfig;
class G4UIdirectory;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;

/**
 * @brief CCIConfigMessenger 类负责创建 Geant4 UI 命令，
 *        并将这些命令与 CCIConfig 单例中的参数设置方法连接起来。
 */
class CCIConfigMessenger : public G4UImessenger
{
public:
    CCIConfigMessenger(CCIConfig* config);
    ~CCIConfigMessenger() override;

    // 当 UI 命令被执行时，此方法会被调用
    void SetNewValue(G4UIcommand* command, G4String newValue) override;

private:
    CCIConfig* fConfig; // 指向 CCIConfig 实例
    G4UIdirectory* fDir;

    // Existing commands
    G4UIcmdWithAnInteger* fDetNXCmd;
    G4UIcmdWithAnInteger* fDetNYCmd;
    G4UIcmdWithADoubleAndUnit* fDetSizeCmd;
    G4UIcmdWithADoubleAndUnit* fDetThickCmd;
    G4UIcmdWithADoubleAndUnit* fDetPitchCmd;
    G4UIcmdWithADoubleAndUnit* fParticleEnergyCmd;

    // New commands for object dimensions
    G4UIcmdWithADoubleAndUnit* fObjectXCmd;
    G4UIcmdWithADoubleAndUnit* fObjectYCmd;
    G4UIcmdWithADoubleAndUnit* fObjectZCmd;

    // New command for detector position
    G4UIcmdWith3VectorAndUnit* fDetCenterCmd;
    G4UIcmdWithADoubleAndUnit* fDetThetaCmd;

    // New commands for slit parameters
    G4UIcmdWithADoubleAndUnit* fSlit2DetDistCmd;
    G4UIcmdWithADoubleAndUnit* fSlitThickCmd;
    G4UIcmdWithADoubleAndUnit* fSlitWidthCmd;
    G4UIcmdWithADoubleAndUnit* fSlitHeightCmd;

    // New commands for primary generator
    G4UIcmdWithADoubleAndUnit* fXStartCmd;
    G4UIcmdWithADoubleAndUnit* fXEndCmd;
    G4UIcmdWithADoubleAndUnit* fZPosCmd;
    G4UIcmdWithADoubleAndUnit* fOmegaYCmd;
    G4UIcmdWithADoubleAndUnit* fVxCmd;
    G4UIcmdWithADoubleAndUnit* fThetaMinCmd;
    G4UIcmdWithADoubleAndUnit* fThetaMaxCmd;
};
