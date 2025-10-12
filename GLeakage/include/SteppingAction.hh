#ifndef STEPPING_ACTION_HH
#define STEPPING_ACTION_HH

#include "G4UserSteppingAction.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "globals.hh"

class G4Step;

/**
 * SteppingAction
 * Ŀ�꣺
 *  - ����ע gamma
 *  - ��׽����໥���ã�compt/phot/Rayl��
 *  - ��׽���������仯����ֵ���裩
 *  - ��ӡ�߽紩Խʱ��ǰ����ϣ������Ŵ�
 */
class SteppingAction : public G4UserSteppingAction {
public:
  explicit SteppingAction(G4double dEthreshold = 10.*eV, G4int verbose = 1);
  ~SteppingAction() override = default;

  void UserSteppingAction(const G4Step* step) override;

  void SetEnergyChangeThreshold(G4double thr) { fDEthr = thr; }
  void SetVerbose(G4int v) { fVerbose = v; }
  void SetMaxPrint(G4int n) { fMaxPrint = n; } // ����ˢ��������ӡǰ n ��������¼��<=0 ��ʾ�����ƣ�

private:
  G4double fDEthr;     // �����仯�ж���ֵ
  G4int    fVerbose;   // 0 ��Ĭ / 1 ��Ҫ / 2 ��ϸ
  G4int    fMaxPrint;  // ����ӡ������ÿ���̣߳�
  G4int    fPrinted;   // �Ѵ�ӡ������ÿ���̣߳�
};

#endif // STEPPING_ACTION_HH
