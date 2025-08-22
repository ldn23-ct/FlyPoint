#include "Config.hh"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

Config& Config::Instance() {
    static Config instance;
    return instance;
}

void Config::ParseCommandLine(int argc, char** argv)
{
    for(int i = 1; i < argc; ++i)
    {
        if(strcmp(argv[i], "-openui") == 0) { openUI = true; }
        else if (strcmp(argv[i], "-runmac") == 0 && i+1 < argc) { RunMacFile = argv[++i]; }
        else if (strcmp(argv[i], "-spectrum") == 0 && i+1 < argc) { SrcSpectrumFile = argv[++i]; }
        else if (strcmp(argv[i], "-result") == 0 && i+1 < argc) { ResultDir = argv[++i]; }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: ./yourApp [-openui] [-runmac filename] [-spectrum filename] [-result resultdir]" << std::endl;
            exit(0);
        }    
    }
}

void Config::InitFromSpectrumFile()
{
    std::ifstream fin(SrcSpectrumFile);
    if(!fin) {
        std::cerr << "Cannot open " << SrcSpectrumFile << std::endl;
        return;
    }
    std::string line;
    std::vector<double> e;

    while ((std::getline(fin, line)))
    {
        std::istringstream iss(line);
        double value;
        if (iss >> value) {
            e.push_back(value);
        }
    }
    fin.close();

    if (e.size() < 2) {
        std::cerr << "Spectrum file format error: less than two lines with numeric data!" << std::endl;
        return;
    }
    Ne = G4int(e.size() - 1);
    e_width = G4double(e[1] - e[0]);
    e_start = G4double(e[0]);
}