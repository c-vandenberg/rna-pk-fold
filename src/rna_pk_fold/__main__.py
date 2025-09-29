from rna_pk_fold.energies.energy_loader import SecondaryStructureEnergyLoader

def main():
    yaml = '/home/chris-vdb/Computational-Chemistry/instadeep/pseudoknots_assessment/rna-pk-fold/data/turner2004_min.yaml'
    secondary_struct_energy_loader = SecondaryStructureEnergyLoader()
    secondary_struct_energy_loader.load(yaml_path=yaml)


if __name__ == '__main__':
    main()