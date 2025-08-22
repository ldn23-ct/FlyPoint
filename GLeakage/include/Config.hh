#ifndef CONFIG_HH
#define CONFIG_HH

#include <string>
#include <globals.hh>
#include "G4ThreeVector.hh"

class Config
{
private:
    Config(): SrcSpectrumFile("../spectrum/160.txt"), RunMacFile(""), ResultDir("../result"), openUI(false),
              Ncos(100), Nphi(100), Ne(205), WorldBoxL(2000), R1(500), R2(1000), e_width(0.001),
              e_start(0), SrcPos(0, 0, 0)
    {}
    std::string SrcSpectrumFile;
    std::string RunMacFile;
    std::string ResultDir;
    bool openUI;

    // data
    G4int Ncos, Nphi, Ne;
    G4double WorldBoxL, R1, R2, e_width, e_start;
    G4ThreeVector SrcPos;

public:
    static Config& Instance();
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
    void ParseCommandLine(int argc, char** argv);
    void InitFromSpectrumFile();
    inline G4String GetSrcSpectrumFile() const { return SrcSpectrumFile; }
    inline G4String GetRunMacFile() const { return RunMacFile; }
    inline G4String GetResultDir() const { return ResultDir; }
    inline G4bool GetOpenUI() const { return openUI; }
    inline G4int GetNcos() const { return Ncos; }
    inline G4int GetNphi() const { return Nphi; }
    inline G4int GetNe() const { return Ne; }
    inline G4double GetWorldBoxL() const { return WorldBoxL; }
    inline G4double GetR1() const { return R1; }
    inline G4double GetR2() const { return R2; }
    inline G4double GetEstart() const { return e_start; }
    inline G4double GetEwidth() const { return e_width; }
    inline G4ThreeVector GetSrcPos() const { return SrcPos; }
};

#endif