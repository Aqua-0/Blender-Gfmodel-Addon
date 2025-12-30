bl_info = {
    "name": "GFModel (GFL2) Importer",
    "author": "Aqua",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "File > Import > GFModel (GFL2) (.gfmodel/.bin/CP/CM)",
    "description": "Import GFL2 GFModel containers (CP/CM) with textures and skeletal animations with Spica/Ohana Refrence",
    "category": "Import-Export",
}

                
                                                                                            
try:
    import bpy                

    from .gfmodel_impl import register, unregister
except ModuleNotFoundError:
                                                                                
    def register() -> None:                          
        raise RuntimeError("This add-on must be registered from within Blender (bpy).")

    def unregister() -> None:                          
        raise RuntimeError(
            "This add-on must be unregistered from within Blender (bpy)."
        )
