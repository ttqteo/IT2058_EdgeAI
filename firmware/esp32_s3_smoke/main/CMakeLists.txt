set(GENERATED_DIR "${CMAKE_CURRENT_LIST_DIR}/../generated")

idf_component_register(
    SRCS "main.cc" "pcm_frontend.cc" "template_store.cc"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp-tflite-micro esp_timer esp_driver_usb_serial_jtag nvs_flash
    EMBED_FILES
        "${GENERATED_DIR}/encoder_int8.tflite"
        "${GENERATED_DIR}/real_train_000_input.int8"
        "${GENERATED_DIR}/real_train_000_expected.int8"
        "${GENERATED_DIR}/real_train_001_input.int8"
        "${GENERATED_DIR}/real_train_001_expected.int8"
        "${GENERATED_DIR}/real_train_002_input.int8"
        "${GENERATED_DIR}/real_train_002_expected.int8"
        "${GENERATED_DIR}/real_train_003_input.int8"
        "${GENERATED_DIR}/real_train_003_expected.int8"
        "${GENERATED_DIR}/real_train_004_input.int8"
        "${GENERATED_DIR}/real_train_004_expected.int8"
)
