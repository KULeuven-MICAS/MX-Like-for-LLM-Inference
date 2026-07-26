# 删掉旧的 sbt source list（如果有）
rm -f /etc/apt/sources.list.d/sbt.list /etc/apt/sources.list.d/sbt_old.list

# 下载并保存 GPG key（现代方式）
curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" \
  | gpg --dearmor \
  | tee /etc/apt/keyrings/sbt.gpg > /dev/null

# 添加 source，指定 signed-by
echo "deb [signed-by=/etc/apt/keyrings/sbt.gpg] https://repo.scala-sbt.org/scalasbt/debian all main" \
  | tee /etc/apt/sources.list.d/sbt.list

mkdir -p /etc/apt/keyrings  # 如果目录不存在先建

apt-get update
apt-get install -y sbt