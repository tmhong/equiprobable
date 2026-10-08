// Generate Pythia8 minimum-bias events and write them straight into a Delphes
// pile-up library (.pileup), using Delphes' own DelphesPileUpWriter.
//
// Why not DelphesPythia8 + root2pileup: on this machine ROOT/Delphes are x86_64
// (Rosetta) while the MG5 HEPTools Pythia8 is arm64, so they cannot be linked into
// one binary.  DelphesPileUpWriter/DelphesXDRWriter do not depend on ROOT, so they
// are compiled here together with Pythia8 (native arm64).
//
// What is written per event = what hepmc2pileup writes from Pythia's HepMC output:
// every final-state particle, PID, production vertex (x, y, z [mm], t [mm/c]) and
// four-momentum (px, py, pz, E [GeV]).
//
// Usage: minbias2pileup <pythia.cmnd> <nevents> <seed> <output.pileup>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "Pythia8/Pythia.h"
#include "classes/DelphesPileUpWriter.h"

int main(int argc, char *argv[])
{
  if(argc != 5)
  {
    std::cerr << "Usage: " << argv[0] << " <pythia.cmnd> <nevents> <seed> <output.pileup>\n";
    return 1;
  }
  const long nEvents = std::atol(argv[2]);
  const int seed = std::atoi(argv[3]);

  Pythia8::Pythia pythia;
  if(!pythia.readFile(argv[1])) return 1;
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + std::to_string(seed));
  if(!pythia.init()) return 1;

  try
  {
    DelphesPileUpWriter writer(argv[4]);
    long written = 0, failed = 0, particles = 0;
    while(written < nEvents)
    {
      if(!pythia.next())
      {
        if(++failed > 100) throw std::runtime_error("too many Pythia8 failures");
        continue;
      }
      const Pythia8::Event &event = pythia.event;
      for(int i = 0; i < event.size(); ++i)
      {
        const Pythia8::Particle &p = event[i];
        if(!p.isFinal()) continue;
        writer.WriteParticle(p.id(),
          p.xProd(), p.yProd(), p.zProd(), p.tProd(),
          p.px(), p.py(), p.pz(), p.e());
        ++particles;
      }
      writer.WriteEntry();
      ++written;
    }
    writer.WriteIndex();
    pythia.stat();
    std::cout << "minbias2pileup: wrote " << written << " events, "
              << double(particles) / written << " particles/event, "
              << failed << " failed Pythia8 events -> " << argv[4] << std::endl;
  }
  catch(std::exception &e)
  {
    std::cerr << "minbias2pileup: ERROR " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
