#include "PhysicsList.hh"

#include "G4EmPenelopePhysics.hh"
#include "G4DecayPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"
#include "G4EmStandardPhysics.hh"

PhysicsList::PhysicsList()
: G4VModularPhysicsList()
{
  SetVerboseLevel(1);

  RegisterPhysics(new G4DecayPhysics);
  RegisterPhysics(new G4RadioactiveDecayPhysics);
  RegisterPhysics(new G4EmPenelopePhysics);
 }
PhysicsList::~PhysicsList()
{ }
void PhysicsList::SetCuts()
{
  G4VUserPhysicsList::SetCuts();
}
