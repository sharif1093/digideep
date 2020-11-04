import pkgutil
package = pkgutil.get_loader("digideep")
print(">> <digideep> is being loaded from:", package.path)
