# External-flash programming with STM32CubeProgrammer

The NUCLEO-N657X0-Q deployment consists of a first-stage bootloader, an application image, and model constants stored in external NOR. Intel HEX files include their destination addresses. Binary application images require an explicit write address.

## Connection and loader selection

1. Connect the ST-LINK interface on CN9 to the workstation.
2. Select development mode with BOOT0/JP1 in position 1 and BOOT1/JP2 in position 2.
3. Open STM32CubeProgrammer, select SWD with Hot Plug mode, and connect.
4. Select the `MX25UM51245G_STM32N6570-NUCLEO.stldr` external loader.

## Image selection

A fresh installation can use either the combined reference HEX or the individual images listed below. The combined reference HEX contains the bootloader, application, and COCO constants. [Combined reference image](../Binary/NUCLEO-N657X0-Q/USB-UVC-Display/NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation-uvc.hex)

| Component | File | Destination |
| --- | --- | --- |
| First-stage bootloader | `FSBL/ai_fsbl.hex` | Encoded base `0x70000000` |
| CubeIDE application | UVCL `*-trusted.bin` | Explicit address `0x70100000` |
| Makefile application | `Project_sign.hex` | Encoded base `0x70100000` |
| Model constants | `Model/NUCLEO-N657X0-Q/network_data.hex` | Encoded base `0x70380000` |

Only one application image is required. Model replacement requires matching generated application code and external constants. The [README](../README.md#5-build-the-application) documents image preparation and the PowerShell programming commands.

## Programming and verification

The Erasing & Programming interface accepts the selected image. Ordinary updates use erase-before-programming for the affected sectors and programming verification. **Skip flash erase before programming** applies when the target region has already been erased. A full-chip erase removes all three components and requires their subsequent restoration.

After programming, both boot jumpers are placed in position 1 and the board is power-cycled. The UART console identifies the application and model. The CN8 USB stream provides the displayed inference results. [Boot sequence](Boot-Overview.md) · [STM32CubeProgrammer manual](https://www.st.com/resource/en/user_manual/dm00403500.pdf)
