def check_protocol_compliance(obj, protocol, verbose: bool = False):
    """Check why an object doesn't match a protocol.
    
    Args:
        obj: The object to check
        protocol: The protocol to check against
        verbose: If True, prints detailed information about the check
    """
    if verbose:
        print(f"\nChecking compliance of {obj.__class__.__name__} with {protocol.__name__}:")
        
        # Print all protocol attributes and methods
        print("\nProtocol contents:")
        for name, value in protocol.__dict__.items():
            if not name.startswith('_'):
                print(f"- {name}: {'Method' if callable(value) else 'Attribute'}")
        
        # Get all required attributes and methods from the protocol
        required_attrs = {
            name: value for name, value in protocol.__annotations__.items()
            if not name.startswith('_')
        }
        
        required_methods = {
            name: value for name, value in protocol.__dict__.items()
            if callable(value) and not name.startswith('_')
        }
        
        print("\nRequired attributes from annotations:")
        for attr_name, attr_type in required_attrs.items():
            print(f"- {attr_name}: {attr_type}")
        
        print("\nRequired methods:")
        for method_name, method in required_methods.items():
            print(f"- {method_name}")
        
        # Check if the protocol is runtime checkable
        if not hasattr(protocol, '_is_runtime_protocol'):
            print("\nWARNING: Protocol is not runtime checkable! Add @runtime_checkable decorator.")
        
        # Check the object's attributes and methods
        print("\nObject's attributes and methods:")
        for name, value in obj.__dict__.items():
            if not name.startswith('_'):
                print(f"- {name}: {'Method' if callable(value) else 'Attribute'}")
    
    # Always print the final result
    result = isinstance(obj, protocol)
    print(f"\n{obj.__class__.__name__} {'matches' if result else 'does not match'} {protocol.__name__}")
    return None