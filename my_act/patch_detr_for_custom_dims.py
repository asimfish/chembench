"""
Patch script to update detr_vae.py to support configurable state/action dimensions.
This is needed when your robot has different dimensions than the default bimanual setup.
"""

import os
import sys

def patch_detr_vae():
    """Patch detr_vae.py to make state_dim configurable via args."""
    
    detr_vae_path = os.path.join(os.path.dirname(__file__), 'detr/models/detr_vae.py')
    
    # Read the file
    with open(detr_vae_path, 'r') as f:
        content = f.read()
    
    # Backup original if not already done
    backup_path = detr_vae_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"Created backup: {backup_path}")
    
    # Check if already patched
    if 'args.state_dim' in content or 'args.action_dim' in content:
        print("detr_vae.py appears to already be patched!")
        return
    
    # Replace hardcoded state_dim in DETRVAE.__init__
    # Line ~58: self.input_proj_robot_state = nn.Linear(14, hidden_dim)
    content = content.replace(
        'self.input_proj_robot_state = nn.Linear(14, hidden_dim)',
        'self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)'
    )
    
    # Line ~61: self.input_proj_robot_state = nn.Linear(14, hidden_dim)
    # Already handled above
    
    # Line ~69: self.encoder_action_proj = nn.Linear(14, hidden_dim)
    content = content.replace(
        'self.encoder_action_proj = nn.Linear(14, hidden_dim) # project action to embedding',
        'self.encoder_action_proj = nn.Linear(self.action_dim, hidden_dim) # project action to embedding'
    )
    
    # Line ~70: self.encoder_joint_proj = nn.Linear(14, hidden_dim)
    content = content.replace(
        'self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding',
        'self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding'
    )
    
    # Update DETRVAE.__init__ to store action_dim
    # Find the __init__ method and add self.action_dim
    old_init_signature = 'def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names):'
    new_init_signature = 'def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, action_dim=None):'
    content = content.replace(old_init_signature, new_init_signature)
    
    # Add action_dim initialization after camera_names
    old_camera_names_line = '        self.camera_names = camera_names'
    new_lines = '''        self.camera_names = camera_names
        self.action_dim = action_dim if action_dim is not None else state_dim'''
    content = content.replace(old_camera_names_line, new_lines)
    
    # Line ~52: self.action_head = nn.Linear(hidden_dim, state_dim)
    content = content.replace(
        'self.action_head = nn.Linear(hidden_dim, state_dim)',
        'self.action_head = nn.Linear(hidden_dim, self.action_dim)'
    )
    
    # CNNMLP class - Line ~169: mlp_in_dim = 768 * len(backbones) + 14
    content = content.replace(
        'mlp_in_dim = 768 * len(backbones) + 14',
        'mlp_in_dim = 768 * len(backbones) + state_dim'
    )
    
    # CNNMLP class - Line ~170: self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
    content = content.replace(
        'self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)',
        'self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=self.action_dim, hidden_depth=2)'
    )
    
    # Update CNNMLP.__init__ to store action_dim
    old_cnnmlp_init = 'def __init__(self, backbones, state_dim, camera_names):'
    new_cnnmlp_init = 'def __init__(self, backbones, state_dim, camera_names, action_dim=None):'
    content = content.replace(old_cnnmlp_init, new_cnnmlp_init)
    
    # Add action_dim to CNNMLP
    old_cnnmlp_state = '        self.state_dim = state_dim'
    new_cnnmlp_state = '''        self.state_dim = state_dim
        self.action_dim = action_dim if action_dim is not None else state_dim'''
    content = content.replace(old_cnnmlp_state, new_cnnmlp_state)
    
    # Update build() function - Line ~230
    old_build_state_dim = '    state_dim = 14 # TODO hardcode'
    new_build_state_dim = '''    state_dim = getattr(args, 'state_dim', 14)
    action_dim = getattr(args, 'action_dim', state_dim)  # Default to state_dim if not specified'''
    content = content.replace(old_build_state_dim, new_build_state_dim)
    
    # Update DETRVAE instantiation in build()
    old_detrvae_create = '''    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )'''
    new_detrvae_create = '''    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        action_dim=action_dim,
    )'''
    content = content.replace(old_detrvae_create, new_detrvae_create)
    
    # Update build_cnnmlp() function - Line ~258
    old_cnnmlp_state_dim = 'def build_cnnmlp(args):\n    state_dim = 14 # TODO hardcode'
    new_cnnmlp_state_dim = '''def build_cnnmlp(args):
    state_dim = getattr(args, 'state_dim', 14)
    action_dim = getattr(args, 'action_dim', state_dim)  # Default to state_dim if not specified'''
    content = content.replace(old_cnnmlp_state_dim, new_cnnmlp_state_dim)
    
    # Update CNNMLP instantiation in build_cnnmlp()
    old_cnnmlp_create = '''    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )'''
    new_cnnmlp_create = '''    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
        action_dim=action_dim,
    )'''
    content = content.replace(old_cnnmlp_create, new_cnnmlp_create)
    
    # Write the patched file
    with open(detr_vae_path, 'w') as f:
        f.write(content)
    
    print(f"Successfully patched {detr_vae_path}")
    print("Changes made:")
    print("  - Made state_dim and action_dim configurable via args")
    print("  - Updated all hardcoded dimension references")
    print("  - Backup saved to", backup_path)

def restore_backup():
    """Restore original detr_vae.py from backup."""
    detr_vae_path = os.path.join(os.path.dirname(__file__), 'detr/models/detr_vae.py')
    backup_path = detr_vae_path + '.backup'
    
    if not os.path.exists(backup_path):
        print("No backup found!")
        return
    
    with open(backup_path, 'r') as f:
        content = f.read()
    
    with open(detr_vae_path, 'w') as f:
        f.write(content)
    
    print(f"Restored {detr_vae_path} from backup")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'restore':
        restore_backup()
    else:
        patch_detr_vae()

