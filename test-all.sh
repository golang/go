#!/bin/bash

# Function to prompt for optional installation
prompt_installation() {
  local name="$1"
  local install_command="$2"
  read -p "Do you want to install $name? (y/n): " INSTALL_CHOICE
  if [[ "$INSTALL_CHOICE" =~ ^[Yy]$ ]]; then
    echo "Installing $name..."
    eval "$install_command"
  else
    echo "$name installation skipped."
  fi
}

# Ask if the user is using Ubuntu or Arch Linux
read -p "Are you using Ubuntu or Arch Linux? (ubuntu/arch): " OS_CHOICE
if [[ "$OS_CHOICE" =~ ^[Uu]buntu$ ]]; then
  INSTALL_CMD="sudo apt install astyle cmake gcc ninja-build libssl-dev python3-pytest python3-pytest-xdist unzip xsltproc doxygen graphviz python3-yaml git pkg-config || exit 1"
elif [[ "$OS_CHOICE" =~ ^[Aa]rch$ ]]; then
  INSTALL_CMD="sudo pacman -S --needed astyle cmake gcc ninja openssl lib32-openssl python-pytest python-pytest-xdist unzip libxslt doxygen graphviz python-yaml git pkgconf || exit 1"
else
  echo "Unsupported OS choice. Exiting."
  exit 1
fi

# Install Liboqs
prompt_installation "Liboqs" "
  $INSTALL_CMD
  git clone https://github.com/open-quantum-safe/liboqs.git || exit 1
  cd liboqs || exit 1
  mkdir -p build && cd build || exit 1
  cmake -GNinja -DBUILD_SHARED_LIBS=ON .. || exit 1
  sudo ninja || exit 1
  sudo ninja install || exit 1
  cd ../..
"

# Install Liboqs-go
prompt_installation "Liboqs-go" "
  git clone https://github.com/open-quantum-safe/liboqs-go.git || exit 1
  export LIBOQS_GO_DIR=$(pwd)/liboqs-go
"

# Set environment variables
echo -e "Setting environment variables..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$LIBOQS_GO_DIR/.config

{
  echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/lib"
  echo "export PKG_CONFIG_PATH=\$PKG_CONFIG_PATH:$LIBOQS_GO_DIR/.config"
} >> ~/.profile
{
  echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/lib"
  echo "export PKG_CONFIG_PATH=\$PKG_CONFIG_PATH:$LIBOQS_GO_DIR/.config"
} >> ~/.bashrc

# Ask user if they want to download the go-std repository
prompt_installation "go-std repository" "
  git clone https://github.com/PQC-Group-UTFPR/go-std.git || {
      echo \"Failed to clone the repository. Exiting.\"; exit 1;
  }
"

# Check if go-std directory exists or ask for its location
if [ ! -d "go-std" ]; then
  read -p "The go-std directory does not exist. Please specify its location (default is '.'): " GO_STD_DIR
  GO_STD_DIR=${GO_STD_DIR:-.}
  if [ ! -d "$GO_STD_DIR" ]; then
    echo "The specified directory does not exist. Exiting."; exit 1;
  fi
  cd "$GO_STD_DIR" || {
    echo "Failed to enter the specified directory. Exiting."; exit 1;
  }
else
  cd go-std || {
    echo "Failed to enter the go-std directory. Exiting."; exit 1;
  }
fi

# Ask user if they want to install the repository
prompt_installation "the repository" "
  cd src || {
      echo \"Failed to enter the src directory. Exiting.\"; exit 1;
  }

  GOOS=linux GOARCH=amd64 ./bootstrap.bash || {
      echo \"Failed to execute bootstrap.bash. Exiting.\"; exit 1;
  }
  ./all.sh || {
      echo \"Failed to execute all.sh. Exiting.\"; exit 1;
  }
  cd - > /dev/null
"

# Ask for test duration
read -p "How many seconds do you want to run the tests? (default is 3): " TEST_DURATION
TEST_DURATION=${TEST_DURATION:-3}

if ! [[ "$TEST_DURATION" =~ ^[0-9]+$ ]]; then
  echo "Invalid input. Please enter a positive integer for the test duration. Exiting."; exit 1;
fi

# Run the tests
echo "Running tests for $TEST_DURATION seconds..."
go test -v ./src/crypto/hybrid -duration=${TEST_DURATION}s || {
  echo "Failed to run the tests. Exiting."; exit 1;
}

echo "Tests completed."
