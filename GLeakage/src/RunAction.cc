#include "RunAction.hh"
#include "Run.hh"
#include "Config.hh"
#include "PrimaryGeneratorAction.hh"
#include "DetectorConstruction.hh"
#include "G4RunManager.hh"
#include "G4Run.hh"
#include "G4AccumulableManager.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleGun.hh"
#include "G4VVisManager.hh"
#include "G4Polyline.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include <sys/stat.h>
#include <sstream>
#include <fstream>
#include <string>

std::vector<G4ThreeVector> RunAction::ReadPos(const std::string& path)
{
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Cannot open file: " + path);
    double x, y, z;
    std::vector<G4ThreeVector> pts;
    pts.reserve(1024);
    while (fin >> x >> y >> z)
    {
        pts.emplace_back(x*mm, y*mm, z*mm);
    }
    return pts;
}


void RunAction::DrawLine()
{
    auto pts = ReadPos("../pos1.txt");
    if (pts.empty()) return;

    auto visManager = G4VVisManager::GetConcreteInstance();
    if (!visManager) return;

    G4VisAttributes va(G4Colour(0.0, 1.0, 0.0));

    const G4ThreeVector src(0.,0.,0.);
    for(const auto& p:pts)
    {
        G4Polyline line;
        line.push_back(src);
        line.push_back(p);
        line.SetVisAttributes(va);
        visManager->Draw(line);
    }
}

void RunAction::BeginOfRunAction(const G4Run*)
{
    DrawLine();
}

void RunAction::EndOfRunAction(const G4Run* run)
{
    if (!G4Threading::IsMasterThread()) return;
    const Run* main_run = static_cast<const Run*>(run);
    int runID = main_run->GetRunID();
    std::string resultDir = Config::Instance().GetResultDir();
#if defined(_WIN32)
  _mkdir(resultDir.c_str());
#else
  mkdir(resultDir.c_str(), 0777); // 目录存在时不报错
#endif
    std::ostringstream CntName;
    CntName << resultDir << "/run" << runID << ".csv";
    G4int row = main_run->Cnts.size();
    G4int col = main_run->Cnts[0].size();
    std::ofstream CntOut(CntName.str());
    for(size_t i = 0; i < row; ++i){
        for(size_t j = 0; j < col; ++j){
            CntOut << main_run->Cnts[i][j];
            if(j < col - 1) CntOut << ",";
        }
        CntOut  << "\n";
    }
    CntOut.close();
}

G4Run* RunAction::GenerateRun()
{
    return new Run(Config::Instance().GetNcos()*Config::Instance().GetNphi(), Config::Instance().GetNe());
}