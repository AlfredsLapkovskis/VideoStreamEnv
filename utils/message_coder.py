from models.messages import *


class MessageCoder:

    _in_message_types = {
        InMessageType.CONNECT: InMessageConnect,
        InMessageType.DISCONNECT: InMessageDisconnect,
        InMessageType.UPDATE_SETTINGS: InMessageUpdateSettings,
        InMessageType.UPDATE_SLOS: InMessageUpdateSlos,
        InMessageType.METRICS: InMessageMetrics,
    }

    _divider = b"_"

    # TODO: - Handle errors
    def decode_in_message(self, raw_bytes: bytes) -> InMessage | None:
        divider_idx = raw_bytes.index(self._divider)
        content_idx = divider_idx + len(self._divider)

        try:
            raw_message_type, raw_content = raw_bytes[:divider_idx], raw_bytes[content_idx:]
            message_type = InMessageType(int.from_bytes(raw_message_type))

            if message_type in self._in_message_types:
                return self._in_message_types[message_type](raw_content)
            else:
                print(f"No inbound message type found: {message_type=}, {raw_bytes=}")
        except Exception as exception:
            print(f"An error occurred while decoding an inbound message: {exception=}, {raw_bytes=}")

        return None


    def encode_out_message(self, message: OutMessage) -> bytes:
        type_bytes = message.type.value.to_bytes(1)
        message_bytes = type_bytes + self._divider + message.encode_content()

        return message_bytes