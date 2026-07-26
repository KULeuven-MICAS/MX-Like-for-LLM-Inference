make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_mx_cluster.hjson rtl-gen
make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_mx_cluster.hjson bin/snitch_cluster.vlt -j
make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_mx_cluster.hjson sw -j
./target/snitch_cluster/bin/snitch_cluster.vlt ./target/snitch_cluster/sw/apps/snax-mx/build/snax-mx.elf