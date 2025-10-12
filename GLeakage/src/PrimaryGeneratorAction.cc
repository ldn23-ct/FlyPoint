#include "PrimaryGeneratorAction.hh"
#include "Config.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4RandomDirection.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include <vector>
#include <string>
#include <fstream>

void PrimaryGeneratorAction::LoadSrcSpectrum(const std::string& filename)
{
    std::ifstream fin(filename.c_str());
    if(!fin) {
        std::cerr << "Cannot open " << filename << std::endl;
        exit(1);
    }

    double e, p;
    double sum = 0;
    std::vector<double> tmp_prob;
    energy.clear();
    cumulative.clear();
    while (fin >> e >> p)
    {
        energy.push_back(e * MeV);
        tmp_prob.push_back(p);
        sum += p;
    }
    double acc = 0;
    for(size_t i = 0; i < tmp_prob.size(); ++i)
    {
        acc += tmp_prob[i]/sum;
        cumulative.push_back(acc);
    }
}

G4double PrimaryGeneratorAction::SampleEnergy()
{
    double r = G4UniformRand();
    for(size_t i=0; i<cumulative.size(); ++i) {
        if(r < cumulative[i])
            return energy[i];
    }
    return energy.back();  
}

void PrimaryGeneratorAction::SetDir()
{
    G4double xtgt = (G4UniformRand() - 0.5) * 16;
    G4double ytgt = (G4UniformRand() - 0.5) * 68;
    G4double ztgt = 30;
    // G4double xtgt = (G4UniformRand() - 0.5) * 5;
    // G4double ytgt = (G4UniformRand() - 0.5) * 30;
    // G4double ztgt = 137;

    G4ThreeVector srcPos(0, 0, 0);
    fParticleGun->SetParticlePosition(srcPos);
    G4ThreeVector tgtPos(xtgt*mm, ytgt*mm, ztgt*mm);
    G4ThreeVector dir = (tgtPos - srcPos).unit();

    fParticleGun->SetParticleMomentumDirection(dir);
}


PrimaryGeneratorAction::PrimaryGeneratorAction()
{
    G4int n_particle = 1;
    fParticleGun  = new G4ParticleGun(n_particle);
    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    G4String particleName;
    G4ParticleDefinition* particle = particleTable->FindParticle(particleName="gamma");
    fParticleGun->SetParticleDefinition(particle);
    LoadSrcSpectrum(Config::Instance().GetSrcSpectrumFile());    
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
    // fParticleGun->SetParticleEnergy(SampleEnergy());
    fParticleGun->SetParticleEnergy(160*keV);
    fParticleGun->SetParticlePosition(Config::Instance().GetSrcPos());
    SetDir();
    // fParticleGun->SetParticleMomentumDirection(G4RandomDirection());
    // fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1*mm));
    fParticleGun->GeneratePrimaryVertex(anEvent);
}