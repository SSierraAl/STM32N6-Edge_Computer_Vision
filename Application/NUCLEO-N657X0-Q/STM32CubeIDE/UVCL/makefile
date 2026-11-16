################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include stm32-vision-models-postprocessing/subdir.mk
-include ll_aton/subdir.mk
-include Utilities/lcd/subdir.mk
-include STM32Cube_FW_N6/Middlewares/ST/usbx/common/subdir.mk
-include Middlewares/stm32-mw-camera/sensors/drivers/subdir.mk
-include Middlewares/stm32-mw-camera/sensors/subdir.mk
-include Middlewares/stm32-mw-camera/subdir.mk
-include Middlewares/screenl/uvcl/subdir.mk
-include Middlewares/screenl/subdir.mk
-include Middlewares/ISP_Library/subdir.mk
-include Drivers/STM32N6xx_HAL_Driver/Src/subdir.mk
-include Drivers/STM32N6xx_HAL_Driver/subdir.mk
-include Drivers/CMSIS/subdir.mk
-include Drivers/BSP/STM32N6xx_Nucleo/subdir.mk
-include Drivers/BSP/Components/subdir.mk
-include Application/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(S_DEPS)),)
-include $(S_DEPS)
endif
ifneq ($(strip $(S_UPPER_DEPS)),)
-include $(S_UPPER_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

OPTIONAL_TOOL_DEPS := \
$(wildcard ../makefile.defs) \
$(wildcard ../makefile.init) \
$(wildcard ../makefile.targets) \


BUILD_ARTIFACT_NAME := NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation
BUILD_ARTIFACT_EXTENSION := elf
BUILD_ARTIFACT_PREFIX :=
BUILD_ARTIFACT := $(BUILD_ARTIFACT_PREFIX)$(BUILD_ARTIFACT_NAME)$(if $(BUILD_ARTIFACT_EXTENSION),.$(BUILD_ARTIFACT_EXTENSION),)

# Add inputs and outputs from these tool invocations to the build variables 
EXECUTABLES += \
NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.elf \

MAP_FILES += \
NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.map \

SIZE_OUTPUT += \
default.size.stdout \

OBJDUMP_LIST += \
NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.list \

OBJCOPY_BIN += \
NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.bin \


# All Target
all:
	+@$(MAKE) --no-print-directory main-build && $(MAKE) --no-print-directory post-build

# Main-build Target
main-build: NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.elf secondary-outputs

# Tool invocations
NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.elf NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.map: $(OBJS) $(USER_OBJS) D:\STM_CODES\STm32\STM32N6-InstanceSegmentation\Application\NUCLEO-N657X0-Q\STM32CubeIDE\STM32N657xx.ld makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-gcc -o "NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.elf" @"objects.list" $(USER_OBJS) $(LIBS) -mcpu=cortex-m55 -T"D:\STM_CODES\STm32\STM32N6-InstanceSegmentation\Application\NUCLEO-N657X0-Q\STM32CubeIDE\STM32N657xx.ld" --specs=nosys.specs -Wl,-Map="NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.map" -Wl,--gc-sections -static -Wl,--cmse-implib -Wl,--out-implib=./secure_nsclib.o -L../../../../Middlewares/stedgeai-lib/Lib/GCC/ARMCortexM55 -Wl,--no-warn-rwx-segments --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -u _printf_float -Wl,--start-group -lc -lm -Wl,--end-group
	@echo 'Finished building target: $@'
	@echo ' '

default.size.stdout: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-size  $(EXECUTABLES)
	@echo 'Finished building: $@'
	@echo ' '

NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.list: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-objdump -h -S $(EXECUTABLES) > "NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.list"
	@echo 'Finished building: $@'
	@echo ' '

NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.bin: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-objcopy  -O binary $(EXECUTABLES) "NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.bin"
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.bin NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.elf NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.list NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.map default.size.stdout
	-@echo ' '

post-build:
	"D:\STM32_PROG\bin\STM32_SigningTool_CLI.exe" -s -bin "NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation.bin" -o "NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation-trusted.bin" -t fsbl -la 0x70100000 -nk -hv 2.3 -align
	-@echo ' '

secondary-outputs: $(SIZE_OUTPUT) $(OBJDUMP_LIST) $(OBJCOPY_BIN)

fail-specified-linker-script-missing:
	@echo 'Error: Cannot find the specified linker script. Check the linker settings in the build configuration.'
	@exit 2

warn-no-linker-script-specified:
	@echo 'Warning: No linker script specified. Check the linker settings in the build configuration.'

.PHONY: all clean dependents main-build fail-specified-linker-script-missing warn-no-linker-script-specified post-build

-include ../makefile.targets
