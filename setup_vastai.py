#!/usr/bin/env python3
"""
Script to automatically set up a vast.ai instance:
1. Sets up SSH keys and config
2. SSHs into the machine
3. Creates ~/.no_auto_tmux file
4. Clones the GitHub repository
5. Installs requirements
"""

import os
import sys
import re
import subprocess
import argparse
from pathlib import Path


def get_or_create_ssh_key(key_name='id_ed25519'):
    """
    Get existing SSH key or create a new one.
    
    Args:
        key_name: Name of the SSH key file (default: id_ed25519)
    
    Returns:
        tuple: (private_key_path, public_key_path)
    """
    ssh_dir = Path.home() / '.ssh'
    ssh_dir.mkdir(exist_ok=True, mode=0o700)
    
    private_key = ssh_dir / key_name
    public_key = ssh_dir / f'{key_name}.pub'
    
    # Check if key exists
    if private_key.exists() and public_key.exists():
        print(f"✓ Found existing SSH key: {private_key}")
        return private_key, public_key
    
    # Generate new key
    print(f"\n⚠️  No SSH key found at {private_key}")
    print("Generating new SSH key...")
    
    # Use ed25519 (modern, secure, fast)
    try:
        result = subprocess.run([
            'ssh-keygen',
            '-t', 'ed25519',
            '-f', str(private_key),
            '-N', '',  # No passphrase for convenience
            '-C', f'vastai-key-{os.getlogin()}'
        ], capture_output=True, text=True, check=True)
        
        # Set proper permissions
        private_key.chmod(0o600)
        public_key.chmod(0o644)
        
        print(f"✓ Generated new SSH key: {private_key}")
        return private_key, public_key
        
    except subprocess.CalledProcessError as e:
        print(f"Error generating SSH key: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def create_ssh_config_entry(host_name, ip, ssh_port, identity_file):
    """
    Create an SSH config entry.
    
    Args:
        host_name: Name for the host (e.g., 'vastai-gpu1')
        ip: IP address
        ssh_port: SSH port number
        identity_file: Path to private key file
    """
    config = f"\nHost {host_name}\n"
    config += f"    HostName {ip}\n"
    config += f"    User root\n"
    config += f"    Port {ssh_port}\n"
    config += f"    IdentityFile {identity_file}\n"
    
    # Add keep-alive settings
    config += "    ServerAliveInterval 60\n"
    config += "    ServerAliveCountMax 3\n"
    
    # Strict host key checking off for vast.ai (IPs change frequently)
    config += "    StrictHostKeyChecking no\n"
    config += "    UserKnownHostsFile=/dev/null\n"
    
    return config


def get_ssh_config_path():
    """Get the path to SSH config file."""
    ssh_dir = Path.home() / '.ssh'
    ssh_dir.mkdir(exist_ok=True, mode=0o700)
    
    config_path = ssh_dir / 'config'
    if not config_path.exists():
        config_path.touch(mode=0o600)
    
    return config_path


def host_exists_in_config(config_path, host_name):
    """Check if a host already exists in the SSH config."""
    if not config_path.exists():
        return False
    
    with open(config_path, 'r') as f:
        content = f.read()
        return re.search(rf'^Host\s+{re.escape(host_name)}\s*$', content, re.MULTILINE) is not None


def remove_host_from_config(config_path, host_name):
    """Remove an existing host from SSH config."""
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    skip = False
    
    for line in lines:
        if line.strip().startswith('Host '):
            # Check if this is the host to remove
            if re.match(rf'Host\s+{re.escape(host_name)}\s*$', line.strip()):
                skip = True
                continue  # Don't add this line
            else:
                skip = False  # Different host, stop skipping
                new_lines.append(line)
        elif not skip:
            # Only add non-Host lines if we're not skipping
            new_lines.append(line)
    
    with open(config_path, 'w') as f:
        f.writelines(new_lines)


def add_to_ssh_config(config_entry, host_name):
    """Add entry to SSH config file."""
    config_path = get_ssh_config_path()
    
    # Check if host already exists
    if host_exists_in_config(config_path, host_name):
        print(f"Host '{host_name}' already exists. Overwriting...")
        # Remove old entry
        remove_host_from_config(config_path, host_name)
    
    # Append new entry
    with open(config_path, 'a') as f:
        f.write(config_entry)
    
    return True


def run_ssh_command(host, command):
    """Run a command on the remote host via SSH."""
    try:
        result = subprocess.run(
            ['ssh', host, command],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running SSH command: {e.stderr}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Set up vast.ai instance: configure SSH, clone repo, and install requirements'
    )
    parser.add_argument('ip_address', help='IP address of the vast.ai instance')
    parser.add_argument('ssh_port', help='SSH port number')
    parser.add_argument('github_repo', help='GitHub repository URL (e.g., https://github.com/user/repo.git)')
    
    args = parser.parse_args()
    
    ip_address = args.ip_address
    ssh_port = args.ssh_port
    github_repo = args.github_repo
    
    # Validate port is a number
    try:
        int(ssh_port)
    except ValueError:
        print("Error: SSH port must be a number.")
        sys.exit(1)
    
    print("=== Vast.ai Setup Script ===\n")
    
    # Step 1: Set up SSH keys
    print("Step 1: Setting up SSH keys...")
    private_key, public_key = get_or_create_ssh_key()
    
    # Display public key for user to add to vast.ai
    with open(public_key, 'r') as f:
        public_key_content = f.read().strip()
    
    print("\n" + "="*60)
    print("📋 YOUR PUBLIC SSH KEY (make sure it's added to vast.ai):")
    print("="*60)
    print(public_key_content)
    print("="*60)
    print("\nIf you haven't added this key to vast.ai yet:")
    print("  1. Go to https://cloud.vast.ai/account/")
    print("  2. Navigate to 'SSH Keys' section")
    print("  3. Click 'Add SSH Key'")
    print("  4. Paste the key above")
    print("="*60 + "\n")
    
    response = input("Have you added this SSH key to vast.ai? (y/n): ").strip().lower()
    if response != 'y':
        print("Please add the SSH key to vast.ai first, then run this script again.")
        sys.exit(1)
    
    # Step 2: Configure SSH connection
    print("\nStep 2: Configuring SSH connection...")
    host_name = "vastai-setup"
    
    print(f"SSH port: {ssh_port}")
    print(f"IP address: {ip_address}")
    print(f"Identity file: {private_key}")
    
    # Generate config entry
    config_entry = create_ssh_config_entry(host_name, ip_address, ssh_port, private_key)
    
    if add_to_ssh_config(config_entry, host_name):
        print(f"\n✓ Successfully added '{host_name}' to SSH config!")
    else:
        print("\nFailed to add to SSH config.")
        sys.exit(1)
    
    # Step 3: SSH into machine and perform setup
    print(f"\nStep 3: Connecting to {ip_address} and setting up...")
    
    # Create ~/.no_auto_tmux file
    print("Creating ~/.no_auto_tmux file...")
    try:
        run_ssh_command(host_name, "touch ~/.no_auto_tmux")
        print("✓ Created ~/.no_auto_tmux")
    except Exception as e:
        print(f"✗ Failed to create ~/.no_auto_tmux: {e}")
        sys.exit(1)
    
    # Extract repo name from GitHub URL
    repo_name = github_repo.rstrip('/').split('/')[-1].replace('.git', '')
    
    # Clone the repository
    print(f"\nCloning repository {github_repo}...")
    clone_command = f"cd ~ && git clone {github_repo} || (cd {repo_name} && git pull)"
    try:
        stdout, stderr = run_ssh_command(host_name, clone_command)
        if stdout:
            print(stdout)
        if stderr and "already exists" not in stderr.lower():
            print(stderr)
        print(f"✓ Repository cloned/updated at ~/{repo_name}")
    except Exception as e:
        print(f"✗ Failed to clone repository: {e}")
        sys.exit(1)
    
    # Install requirements
    print(f"\nInstalling requirements from ~/{repo_name}/requirements.txt...")
    install_command = f"cd ~/{repo_name} && pip install -r requirements.txt"
    try:
        stdout, stderr = run_ssh_command(host_name, install_command)
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        print("✓ Requirements installed successfully")
    except Exception as e:
        print(f"✗ Failed to install requirements: {e}")
        print("Note: This might be expected if requirements.txt doesn't exist or pip install had warnings.")
        # Don't exit on pip install failure, as it might just be warnings
    
    print("\n" + "="*60)
    print("✓ Setup complete!")
    print("="*60)
    print(f"\nYou can now connect with:")
    print(f"  ssh {host_name}")
    print(f"\nRepository is located at: ~/{repo_name}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)

