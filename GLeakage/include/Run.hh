#ifndef RUN_HH
#define RUN_HH

#include "G4Run.hh"
#include <vector>

class Run: public G4Run
{
public:
    std::vector<std::vector<int>> Cnts;
    Run(size_t ndir, size_t ne): Cnts(ndir, std::vector<int>(ne, 0)) {}

    inline void AddCnts(size_t idir, size_t ie) { Cnts[idir][ie]++; }

    void Merge(const G4Run* run) override
    {
        const Run* local = static_cast<const Run*>(run);
        if (Cnts.size() != local->Cnts.size()) {
            G4Exception("Run::Merge", "RunError", FatalException, "Direction bin count mismatch in merge.");
        }
        for(size_t i = 0; i < Cnts.size(); ++i){
            if (Cnts[i].size() != local->Cnts[i].size()) {
                G4Exception("Run::Merge", "RunError", FatalException, "Energy bin count mismatch in merge.");
            }
            for(size_t j = 0; j < Cnts[i].size(); ++j)
                Cnts[i][j] += local->Cnts[i][j];
        }
    }
};

#endif