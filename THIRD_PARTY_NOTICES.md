# Third-party attributions

AerialNav's original project code is offered under BSD-3-Clause. Derivative components and external assets retain their upstream notices and licenses; this document does not relicense third-party work.

| Component | Original source | License and local notice | Scope |
|---|---|---|---|
| Isaac Lab / ORBIT terrain interfaces and structure | https://github.com/isaac-sim/IsaacLab | BSD-3-Clause, `LICENSE-IsaacLab` | Simulator integration and terrain helpers |
| Original NavRL task definitions, navigation references and PPO implementation | https://github.com/Zhefan-Xu/NavRL | MIT, `LICENSE-NavRL` | Model/task reference and PPO baseline |
| OmniDrones | https://github.com/btx0424/OmniDrones | MIT, `LICENSE-OmniDrones` | RotorGroup and Hummingbird robot asset |
| SRU-LSTM-Gated | https://github.com/leggedrobotics/sru-pytorch-spatial-learning | MIT, `LICENSE-SRU` | Optional gated recurrent cell |

The file `src/aerialnav/assets/hummingbird.usd` was byte-for-byte compared against
`isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/hummingbird.usd`
from the available NavRL third-party checkout. Both had SHA-256:

`b4391716dd2c3636c5f82436d09bd72d38daf249d546f927da22e553dbf20bee`

Proper upstream names in this attribution and in the accompanying license texts are intentional. They are not current project package, experiment, or method identifiers.

Isaac Sim and other binary dependencies may have additional terms beyond this project's source license. Consult NVIDIA's license terms before redistributing their binaries; this repository does not vendor the simulator.

cuRoboV2 is an external dependency from https://github.com/NVlabs/curobo, pinned to revision `78fd485fa82d9b9a063fb4985e371814587e666a`. It retains NVIDIA's Apache-2.0 license. The project calls its ESDF and shape-query interfaces; it does not vendor its optimizer or simulator binaries. See `LICENSE-cuRobo`.
