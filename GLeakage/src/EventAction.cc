#include "EventAction.hh"
#include "RunAction.hh"
#include "AirSD.hh"
#include "G4Event.hh"

EventAction::EventAction(RunAction* runAction): fRunAction(runAction)
{}

void EventAction::BeginOfEventAction(const G4Event*) {}

void EventAction::EndOfEventAction(const G4Event* event)
{}
