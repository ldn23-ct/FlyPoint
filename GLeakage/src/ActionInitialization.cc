#include "ActionInitialization.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "SteppingAction.hh"

void ActionInitialization::BuildForMaster() const
{
  auto runAction = new RunAction;
  SetUserAction(runAction);
}

void ActionInitialization::Build() const
{
  SetUserAction(new PrimaryGeneratorAction);

  auto runAction = new RunAction;
  SetUserAction(runAction);

  auto eventAction = new EventAction(runAction);
  SetUserAction(eventAction);

  auto* stepping = new SteppingAction(10.*eV /*�����仯��ֵ*/, 2 /*verbose*/);
  stepping->SetMaxPrint(200); // ��ӡ���ޣ�����ˢ��
  SetUserAction(stepping);
}