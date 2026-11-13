# `swarm-torch-transport-iroh`

Workspace-excluded (`exclude = ["adapters/*"]`) QUIC / Iroh transport carrier adapter implementing `ALPN_SWARMTORCH_DELIVERY_V1` (`b"swarmtorch/delivery/v1"`), mandatory 1-RTT handshake enforcement (rejecting 0-RTT/Half-RTT early data), length-prefixed framing (`QuicLengthPrefixedCodecV1`), and clean separation of transport link principal (`EndpointId`) from application originator identity (`PeerId`).
