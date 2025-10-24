#!/usr/bin/env python3
"""
Requirements Verification Script

Verifies that all required dependencies are installed and compatible
for the Market Edge Finder Experiment project.
"""

import sys
import subprocess
import importlib
import pkg_resources
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define critical dependencies and their minimum versions
CRITICAL_DEPENDENCIES = {
    'torch': '2.1.0',
    'lightgbm': '4.1.0',
    'pandas': '2.1.0',
    'numpy': '1.24.0',
    'v20': '3.0.25.0',  # Official OANDA v20 library
    'pyarrow': '14.0.0',
    'aiohttp': '3.9.0',
    'pyyaml': '6.0',
    'tqdm': '4.65.0',
}

# Optional dependencies
OPTIONAL_DEPENDENCIES = {
    'matplotlib': '3.7.0',
    'plotly': '5.17.0',
    'seaborn': '0.12.0',
    'pytest': '7.4.0',
    'black': '23.9.0',
    'mypy': '1.6.0',
}

# Python version requirements
MIN_PYTHON_VERSION = (3, 11)


def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    current_version = sys.version_info[:2]
    
    if current_version >= MIN_PYTHON_VERSION:
        logger.info(f"✅ Python {sys.version.split()[0]} meets requirements (>= {'.'.join(map(str, MIN_PYTHON_VERSION))})")
        return True
    else:
        logger.error(f"❌ Python {sys.version.split()[0]} is too old (>= {'.'.join(map(str, MIN_PYTHON_VERSION))} required)")
        return False


def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string into tuple for comparison."""
    try:
        # Handle pre-release versions by stripping everything after the first non-digit/dot character
        clean_version = ""
        for char in version_str:
            if char.isdigit() or char == '.':
                clean_version += char
            else:
                break
        
        return tuple(map(int, clean_version.split('.')))
    except ValueError:
        logger.warning(f"Could not parse version: {version_str}")
        return (0,)


def check_package_installed(package_name: str, min_version: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a package is installed and meets minimum version requirements.
    
    Returns:
        Tuple of (is_compatible, installed_version)
    """
    try:
        # Try to import the package
        if package_name == 'v20':
            # Special handling for v20 library
            import v20
            installed_version = v20.__version__
        else:
            module = importlib.import_module(package_name)
            installed_version = getattr(module, '__version__', None)
            
            # Fallback to pkg_resources if __version__ not available
            if not installed_version:
                try:
                    installed_version = pkg_resources.get_distribution(package_name).version
                except pkg_resources.DistributionNotFound:
                    return False, None
        
        if not installed_version:
            logger.warning(f"⚠️ {package_name} installed but version not detectable")
            return True, "unknown"
        
        # Compare versions
        installed_tuple = parse_version(installed_version)
        required_tuple = parse_version(min_version)
        
        is_compatible = installed_tuple >= required_tuple
        
        if is_compatible:
            logger.info(f"✅ {package_name} {installed_version} >= {min_version}")
        else:
            logger.error(f"❌ {package_name} {installed_version} < {min_version} (upgrade required)")
        
        return is_compatible, installed_version
        
    except ImportError:
        logger.error(f"❌ {package_name} not installed")
        return False, None
    except Exception as e:
        logger.error(f"❌ Error checking {package_name}: {str(e)}")
        return False, None


def check_oanda_v20_specifically() -> bool:
    """Special check for OANDA v20 library to ensure it's the official one."""
    try:
        import v20
        
        # Check that it has the expected API structure
        required_attrs = ['Context', 'api', 'account', 'instrument', 'pricing']
        
        for attr in required_attrs:
            if not hasattr(v20, attr):
                logger.error(f"❌ v20 library missing expected attribute: {attr}")
                return False
        
        # Test basic context creation (without actual API call)
        try:
            context = v20.Context(
                hostname="api-fxpractice.oanda.com",
                port=443,
                ssl=True,
                application="VerificationTest",
                token="dummy_token"
            )
            logger.info("✅ OANDA v20 library structure verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ v20 library context creation failed: {str(e)}")
            return False
            
    except ImportError:
        logger.error("❌ OANDA v20 library (v20) not installed")
        return False


def check_pytorch_backend() -> Dict[str, bool]:
    """Check PyTorch backend availability."""
    backends = {
        'cpu': False,
        'cuda': False,
        'mps': False  # Apple Metal Performance Shaders
    }
    
    try:
        import torch
        
        # CPU is always available
        backends['cpu'] = True
        logger.info("✅ PyTorch CPU backend available")
        
        # Check CUDA
        if torch.cuda.is_available():
            backends['cuda'] = True
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            logger.info(f"✅ PyTorch CUDA backend available ({device_count} device(s): {device_name})")
        else:
            logger.info("ℹ️ PyTorch CUDA backend not available")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            backends['mps'] = True
            logger.info("✅ PyTorch MPS (Metal) backend available")
        else:
            logger.info("ℹ️ PyTorch MPS backend not available")
            
    except ImportError:
        logger.error("❌ PyTorch not installed")
    
    return backends


def check_data_directories() -> bool:
    """Check that required data directories exist or can be created."""
    project_root = Path(__file__).parent.parent
    required_dirs = [
        'data',
        'models',
        'logs',
        'results',
        'configs'
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        
        if dir_path.exists():
            logger.info(f"✅ Directory exists: {dir_path}")
        else:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"❌ Cannot create directory {dir_path}: {str(e)}")
                all_good = False
    
    return all_good


def check_environment_variables() -> bool:
    """Check for required environment variables."""
    import os
    
    required_vars = ['OANDA_API_KEY', 'OANDA_ACCOUNT_ID']
    optional_vars = ['OANDA_ENVIRONMENT']
    
    all_good = True
    
    for var in required_vars:
        if os.getenv(var):
            logger.info(f"✅ Environment variable {var} is set")
        else:
            logger.error(f"❌ Environment variable {var} not set")
            all_good = False
    
    for var in optional_vars:
        if os.getenv(var):
            logger.info(f"✅ Optional environment variable {var} is set: {os.getenv(var)}")
        else:
            logger.info(f"ℹ️ Optional environment variable {var} not set (will use default)")
    
    return all_good


def install_missing_packages(missing_packages: List[str]) -> bool:
    """Attempt to install missing packages."""
    if not missing_packages:
        return True
    
    logger.info(f"Attempting to install missing packages: {', '.join(missing_packages)}")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--upgrade'
        ] + missing_packages)
        
        logger.info("✅ Installation completed")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Installation failed: {str(e)}")
        return False


def main():
    """Main verification function."""
    logger.info("🔍 Starting Market Edge Finder requirements verification...\n")
    
    overall_status = True
    
    # Check Python version
    logger.info("=" * 50)
    logger.info("PYTHON VERSION CHECK")
    logger.info("=" * 50)
    if not check_python_version():
        overall_status = False
    
    # Check critical dependencies
    logger.info("\n" + "=" * 50)
    logger.info("CRITICAL DEPENDENCIES CHECK")
    logger.info("=" * 50)
    
    missing_critical = []
    incompatible_critical = []
    
    for package, min_version in CRITICAL_DEPENDENCIES.items():
        is_compatible, installed_version = check_package_installed(package, min_version)
        
        if installed_version is None:
            missing_critical.append(f"{package}>={min_version}")
        elif not is_compatible:
            incompatible_critical.append(f"{package}>={min_version}")
    
    # Check optional dependencies
    logger.info("\n" + "=" * 50)
    logger.info("OPTIONAL DEPENDENCIES CHECK")
    logger.info("=" * 50)
    
    missing_optional = []
    
    for package, min_version in OPTIONAL_DEPENDENCIES.items():
        is_compatible, installed_version = check_package_installed(package, min_version)
        
        if installed_version is None:
            missing_optional.append(f"{package}>={min_version}")
    
    # Special OANDA v20 check
    logger.info("\n" + "=" * 50)
    logger.info("OANDA V20 LIBRARY VERIFICATION")
    logger.info("=" * 50)
    if not check_oanda_v20_specifically():
        overall_status = False
    
    # PyTorch backend check
    logger.info("\n" + "=" * 50)
    logger.info("PYTORCH BACKEND CHECK")
    logger.info("=" * 50)
    backends = check_pytorch_backend()
    
    # Directory structure check
    logger.info("\n" + "=" * 50)
    logger.info("DIRECTORY STRUCTURE CHECK")
    logger.info("=" * 50)
    if not check_data_directories():
        overall_status = False
    
    # Environment variables check
    logger.info("\n" + "=" * 50)
    logger.info("ENVIRONMENT VARIABLES CHECK")
    logger.info("=" * 50)
    if not check_environment_variables():
        logger.warning("⚠️ Missing environment variables - configure before running")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 50)
    
    if missing_critical:
        logger.error("❌ Missing critical packages:")
        for pkg in missing_critical:
            logger.error(f"   - {pkg}")
        overall_status = False
    
    if incompatible_critical:
        logger.error("❌ Incompatible critical packages (need upgrade):")
        for pkg in incompatible_critical:
            logger.error(f"   - {pkg}")
        overall_status = False
    
    if missing_optional:
        logger.warning("⚠️ Missing optional packages:")
        for pkg in missing_optional:
            logger.warning(f"   - {pkg}")
    
    # Installation recommendations
    if missing_critical or incompatible_critical:
        logger.info("\n📦 To install/upgrade required packages, run:")
        if missing_critical:
            logger.info(f"   pip install {' '.join(missing_critical)}")
        if incompatible_critical:
            logger.info(f"   pip install --upgrade {' '.join(incompatible_critical)}")
    
    if missing_optional:
        logger.info("\n📦 To install optional packages, run:")
        logger.info(f"   pip install {' '.join(missing_optional)}")
    
    # Final status
    if overall_status:
        logger.info("\n🎉 All critical requirements verified successfully!")
        logger.info("The Market Edge Finder system is ready to run.")
        return 0
    else:
        logger.error("\n💥 Requirements verification failed!")
        logger.error("Please install missing dependencies before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())