#ifndef CCIDetectorConstruction_h
#define CCIDetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"
class G4VPhysicalVolume;
class G4GlobalMagFieldMessenger;

class CCIDetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    CCIDetectorConstruction();
    virtual ~CCIDetectorConstruction();

  public:
    virtual G4VPhysicalVolume* Construct();
    virtual void ConstructSDandField();

    // get methods
    //
    const G4VPhysicalVolume* GetAbsorberPV() const;
    const G4VPhysicalVolume* GetScatterPV() const;

  private:
    // methods
    //
    void DefineMaterials();
    G4VPhysicalVolume* DefineVolumes();
    void SetInitialValue();

  //  DetectorMessenger*  fMessenger;
    G4double user_Angle;
    G4int  fIsTarget;  //put target or not
    G4double user_tk;
    G4double worldSize;     //the size of the world

    // data members
    //
    static G4ThreadLocal G4GlobalMagFieldMessenger*  fMagFieldMessenger;
                                      // magnetic field messenger

    G4VPhysicalVolume*   fAbsorberPV; // the absorber physical volume
    G4VPhysicalVolume*   fScatterPV;  // the scatter physical volume

    G4bool  fCheckOverlaps; // option to activate checking of volumes overlaps
};

// inline functions

inline const G4VPhysicalVolume* CCIDetectorConstruction::GetAbsorberPV() const {
  return fAbsorberPV;
}
inline const G4VPhysicalVolume* CCIDetectorConstruction::GetScatterPV() const {
  return fScatterPV;
}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
