# Boot sequence and memory allocation

The NUCLEO-N657X0-Q application supports SRAM execution during debugging and autonomous execution following a boot from external flash. The selected boot mode determines how the application reaches its execution address.

## Development mode

In development mode, BOOT0/JP1 is in position 1 and BOOT1/JP2 is in position 2. The debugger can load the application ELF into SRAM. The generated network still references its constants in external NOR, so the corresponding model data must already be programmed at `0x70380000`.

## Boot from external flash

For autonomous operation, both boot jumpers are in position 1. The boot ROM loads the first-stage bootloader, which then loads the application for execution in SRAM. The application initializes the external NOR interface in memory-mapped mode before invoking the network.

| External-flash region | Image |
| --- | --- |
| `0x70000000–0x700FFFFF` | `FSBL/ai_fsbl.hex` |
| `0x70100000–0x7037FFFF` | Application image with STM32 boot header |
| `0x70380000–0x73FFFFFF` | Generated network constants |

The flash address specifies the storage location of an image. The firmware linker script specifies its execution placement. The application image and model constants must originate from compatible generated files. The [programming procedure](Program-Hex-Files-STM32CubeProgrammer.md) describes image selection and loader configuration.

## SRAM allocation

The application linker region begins at `0x34000400` with a length of 1647 KiB, ending at `0x3419C000` as an exclusive bound. The model memory pool reserves a 400 KiB CPU SRAM region beginning at that boundary, together with NPU SRAM banks for activation placement.

The linker script and memory-pool description are separate inputs to separate compilers. Their address ranges must remain coordinated to prevent overlapping allocations. The linker map and generation report provide the resulting placements. [Firmware linker script](../Application/NUCLEO-N657X0-Q/STM32CubeIDE/STM32N657xx.ld#L14) · [Neural-ART memory pool](../Model/my_mpools/stm32n6-app2_NUCLEO-N657X0-Q.mpool)
