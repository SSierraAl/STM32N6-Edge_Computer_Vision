# Neural-ART execution and memory placement

The STM32N6 Neural-ART accelerator executes supported neural network operations under the coordination of the Cortex-M55 application and ST runtime. ST Edge AI Core compiles the graph into a scheduled execution plan and model-specific code. CPU operations and synchronization remain part of the complete inference path.

## Execution artifacts

| Artifact | Function |
| --- | --- |
| `network.c` and `network_ecblobs.h` | Execution plan, generated operations, and accelerator control data |
| `stai_network.c` and `stai_network.h` | Application-facing API and tensor definitions |
| `network_atonbuf.xSPI2.raw` | Model constants assigned to external NOR |
| `network_generate_report.txt` | Graph, memory allocation, and execution-placement information |

The application obtains input and output addresses from the generated API and invokes `stai_network_run()`. The wrapper repeats asynchronous calls while inference remains active and waits for hardware events when requested by the runtime. The generated epoch table specifies units of scheduled execution. The generation report identifies hardware, software, and hybrid epochs, which helps relate latency to operation placement. [Application inference wrapper](../Application/NUCLEO-N657X0-Q/Src/main.c#L302) · [Generated epoch table](../Model/NUCLEO-N657X0-Q/network.c#L2265)

## Weights, activations, and buffers

Weights and other constants remain fixed during inference. Activations hold intermediate results for a particular input. The compiler places these arrays according to access requirements, available capacity, and the memory-pool description. In this project, constants occupy external NOR, while activation storage uses the configured SRAM pools.

The firmware linker independently allocates application code, globals, stack, heap, and display buffers. Its regions must be coordinated with the Neural-ART pools to prevent conflicting allocations. Network input and output addresses also participate in this allocation, so their availability follows the generated execution plan and buffer lifetimes. [Memory pool](../Model/my_mpools/stm32n6-app2_NUCLEO-N657X0-Q.mpool) · [Firmware linker script](../Application/NUCLEO-N657X0-Q/STM32CubeIDE/STM32N657xx.ld)

The CPU supplies or receives data through shared buffers accessed by other hardware blocks. Cache maintenance ensures that producers and consumers observe the intended memory contents. Input capture completes before inference begins, and postprocessing reads the outputs after network execution. Changes to buffering or concurrency require analysis of buffer ownership, allocation lifetime, and cache operations.

## Compiler and runtime compatibility

Generated C/H files and external constants describe one compiled graph. They are updated together when the model is regenerated. The application links against `Middlewares/stedgeai-lib/`, whose API and runtime must be compatible with the generator version. The committed model report identifies ST Edge AI Core 3.0.0.

The network report and firmware linker map provide complementary evidence of placement. The reference network uses 445 of 448 KiB in one NPU SRAM bank. Capacity in other banks does not automatically resolve a placement constraint. [Generation report](../Model/st_ai_output/network_generate_report.txt#L720)

The [architecture description](Application-Overview.md) develops the tensor and memory examples. ST's compiler documentation provides further detail on scheduling and allocation. [ST Edge AI Core documentation](https://stedgeai-dc.st.com/assets/embedded-docs/index.html)
