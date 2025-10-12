#ifndef PRIMARYGENERATORACTION_HH
#define PRIMARYGENERATORACTION_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"
#include <vector>
#include <string>

class G4ParticleGun;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
    PrimaryGeneratorAction();
    ~PrimaryGeneratorAction() override;
    void LoadSrcSpectrum(const std::string& filename);
    void GeneratePrimaries(G4Event*) override;
    const G4ParticleGun* GetParticleGun() const { return fParticleGun; }

private:
    G4double SampleEnergy();
    void SetDir();
    G4ParticleGun* fParticleGun = nullptr; // pointer a to G4 gun class
    std::vector<G4double> energy;
    std::vector<double> cumulative;
};

#endif