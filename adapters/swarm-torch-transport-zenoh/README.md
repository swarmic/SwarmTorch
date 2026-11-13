# `swarm-torch-transport-zenoh`

Workspace-excluded (`exclude = ["adapters/*"]`) Zenoh pub/sub topic carrier adapter enforcing canonical key expression contracts (`swarmtorch/v1/<scope_hex>/round/<round_id>/peer/<originator_hex>/update`), rejecting key-expression wildcards/ROS 2 prefixes/non-canonical integers, and verifying topic-vs-envelope identity agreement before downstream replay or coordinator admission.
