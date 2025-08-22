#include "ActionInitialization.hh"
#include "Config.hh"
#include "DetectorConstruction.hh"
#include "RunAction.hh"
#include "G4RunManagerFactory.hh"
#include "G4SteppingVerbose.hh"
#include "G4UImanager.hh"
#include "QBBC.hh"
#include "FTFP_BERT.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "Randomize.hh"
#include "time.h"

#include <chrono>
#include <iostream>

int main(int argc, char** argv)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    //random engine
    CLHEP::HepRandom::setTheEngine(new CLHEP::RanecuEngine());

    //set random seed
    G4long seed = time(NULL);
    CLHEP::HepRandom::setTheSeed(seed);

    // argv setting
    Config::Instance().ParseCommandLine(argc, argv);
    Config::Instance().InitFromSpectrumFile();
    const Config& config = Config::Instance();

    G4UIExecutive* ui = nullptr;
    if ( config.GetOpenUI() == true ) { ui = new G4UIExecutive(argc, argv); }

    G4int precision = 4;
    G4SteppingVerbose::UseBestUnit(precision);

    // auto runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default);
    auto runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::SerialOnly);
    
    runManager->SetUserInitialization(new DetectorConstruction());

    // auto physicsList = new QBBC;
    // physicsList->SetVerboseLevel(1);
    // runManager->SetUserInitialization(physicsList);

    auto physicsList = new FTFP_BERT;
    physicsList->ReplacePhysics(new G4EmStandardPhysics_option4());
    runManager->SetUserInitialization(physicsList);


    runManager->SetUserInitialization(new ActionInitialization());
    auto visManager = new G4VisExecutive();
    visManager->Initialize();
    auto UImanager = G4UImanager::GetUIpointer();

    if ( ! ui ) {
        G4String command = "/control/execute ";
        G4String fileName = config.GetRunMacFile();
        UImanager->ApplyCommand(command+fileName);
    }
    else {
        UImanager->ApplyCommand("/control/execute init_vis.mac");
        ui->SessionStart();
        delete ui;
    }

    delete visManager;
    delete runManager;

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;
    std::cout << "Total execution time: " << elapsed.count() << " seconds." << std::endl;
    return 0;
}





