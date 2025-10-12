#ifndef STEPPING_ACTION_HH
#define STEPPING_ACTION_HH

#include "G4UserSteppingAction.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "globals.hh"

class G4Step;

/**
 * SteppingAction
 * 目标：
 *  - 仅关注 gamma
 *  - 捕捉电磁相互作用（compt/phot/Rayl）
 *  - 捕捉显著能量变化（阈值可设）
 *  - 打印边界穿越时的前后材料，辅助排错
 */
class SteppingAction : public G4UserSteppingAction {
public:
  explicit SteppingAction(G4double dEthreshold = 10.*eV, G4int verbose = 1);
  ~SteppingAction() override = default;

  void UserSteppingAction(const G4Step* step) override;

  void SetEnergyChangeThreshold(G4double thr) { fDEthr = thr; }
  void SetVerbose(G4int v) { fVerbose = v; }
  void SetMaxPrint(G4int n) { fMaxPrint = n; } // 避免刷屏：最多打印前 n 条触发记录（<=0 表示不限制）

private:
  G4double fDEthr;     // 能量变化判定阈值
  G4int    fVerbose;   // 0 静默 / 1 简要 / 2 详细
  G4int    fMaxPrint;  // 最大打印条数（每个线程）
  G4int    fPrinted;   // 已打印条数（每个线程）
};

#endif // STEPPING_ACTION_HH
