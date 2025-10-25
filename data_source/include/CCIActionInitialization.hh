#ifndef CCIActionInitialization_h
#define CCIActionInitialization_h 1

#include "G4VUserActionInitialization.hh"

class CCIDetectorConstruction;

/// Action initialization class.
///

class CCIActionInitialization : public G4VUserActionInitialization
{
  public:
    CCIActionInitialization(CCIDetectorConstruction*);
    virtual ~CCIActionInitialization();

    virtual void BuildForMaster() const;
    virtual void Build() const;

  private:
    CCIDetectorConstruction* fDetConstruction;
};

#endif
