from pymatgen.ext.matproj import MPRester
import json
from typing import List
import tqdm


class dataScraper:

    """
    Retrieves the compounds for a pool of elements then
    extracts, converts and formats xrd with corresponding space group and lattice parameters
    """

    def __init__(self,
                 pool: List[str],
                 api_key: str,
                 json_path: str,
                 json_name: str,
                 sources: List[str] = None,
                 ):

        """
         :parameter
         pool: List of strings which contains the short forms of elements
         api_key: private api key provided with materials project account
         json_path: path of folder to store json file
         json_name: name of json file that stores final dataset
         """

        self.mp_entries = None
        self.data_dict = {}
        if sources is None:
            sources = ['xrd.Cu', 'xrd.Mo', 'xrd.Fe', 'xrd.Ag']

        self.sources = sources
        self.pool = pool
        self.api_key = api_key

        self.entry_finder()

        self.data_generator()

        with open(json_path + json_name, 'w') as fp:
            json.dump(self.data_dict, fp)

    def entry_finder(self) -> None:

        """Gets all possible combinations of compounds present in repository.

        :parameter


        :returns mp_entries: PyMatGen ComputedEntries of compounds in repository
        """

        with MPRester(self.api_key) as m:
            self.mp_entries = m.get_entries_in_chemsys(self.pool)

    def xrd_retriever(self,
                      entry_id: str,
                      source: str) -> list:

        """Extracts the peak list for a given source

        :parameter
            entry_id: A string which is the id of the crystals in Materials Project
            source: A string in the form of Xrd.X (X to be replaced by Cu, Mo etc)

        :returns Peak list of XRD pattern for given source"""

        properties = [source]

        with MPRester(self.api_key) as m:
            results = m.query(entry_id, properties=properties)

        if results[0][source] is None:  # If the source XRD doesn't exist for one compound
            results = None

        return results

    @staticmethod
    def peak_list_extractor(xrd: list,
                            source: str) -> List[list]:

        """Static method for getting a (2theta, intensity) pair from the peak list

        :parameter
            xrd: peak list for xrd of a given xrd source
            source: xrd source provided
        :returns
            List of corresponding 2theta and intensity lists"""

        patternList = xrd[0][source]['pattern']

        two_theta, intensity = [], []
        for j in patternList:
            two_theta.append(j[2])
            intensity.append(j[0])  # indices taken from standard format from Materials Project

        return [two_theta, intensity]

    def spacegroup_extractor(self,
                             entry_id: str) -> str:

        """Extracts space group for a compound. Is the target property for this project

        :parameter
            entry_id: A string which is the id of the crystals in Materials Project
        :returns
            space group symbol as a string
        """

        properties = ['spacegroup']

        with MPRester(self.api_key) as m:
            results = m.query(entry_id, properties=properties)

        return results[0]['spacegroup']['symbol']

    def lattice_extractor(self,
                          entry_id: str) -> List[float]:

        """Extracts the lattice parameters for a given crystal
        :parameter
            entry_id: A string which is the id of the crystals in Materials Project
        :returns
            list of 6 lattice parameters as float
            """
        properties = ['initial_structure']

        with MPRester(self.api_key) as m:
            results = m.query(entry_id, properties=properties)

        lattice_parameters = results[0]['initial_structure'].as_dict()['lattice']

        return [lattice_parameters['a'], lattice_parameters['b'], lattice_parameters['c'],
                lattice_parameters['alpha'], lattice_parameters['beta'], lattice_parameters['gamma']]

    def targets_saver(self,
                      entry_id: str) -> dict:
        """Formats the lattice and space group as a dictionary for targets
        :parameter
            entry_id: A string which is the id of the crystals in Materials Project
        :returns dictionary with space group and lattice parameters"""
        return {'lattice': self.lattice_extractor(entry_id), 'spacegroup': self.spacegroup_extractor(entry_id)}

    @staticmethod
    def xrd_processor(peak_list: list) -> List[float]:

        """Converts the peak_list to defined size 1D vector where each element in the list corresponds to the intensity
        at that 2theta angle. Chosen to 1800 here, which corresponds to a 0.1 degree step size for [0,180] range.

        :parameter peak_list: List of corresponding 2theta and intensity lists """

        plot_int = peak_list
        vec = []
        vec_size = 1800  # !!!!!!!!!!!!!! The size decided for this project (10 times 180 degrees) !!!!!!!!!!!!!!!!

        for idx, j in enumerate(plot_int[0]):
            plot_int[0][idx] = round(j, 1)

        for i in range(0, vec_size, 1):

            i = i / 10
            # Should find a better way than try-except blocks
            try:

                idx = plot_int[0].index(i)
                vec.append(plot_int[1][idx])

            except:

                vec.append(0.0)

        return vec

    def data_generator(self) -> None:

        """
        The main function which extracts, converts and formats xrd for the full pool of
        compounds for all specified sources

        Formats the data into a dict in the form
        {'X': dictionary of xrd vectors from various sources,
        'Y': dictionary of targets}

         """

        for element, index in tqdm(enumerate(self.mp_entries)):

            xrd_dict = {}

            for source in self.sources:

                xrd = self.xrd_retriever(entry_id=element.entry_id,
                                         source=source)
                if xrd is None:
                    continue

                xrd_vec = self.xrd_processor(self.peak_list_extractor(xrd=xrd, source=source))

                xrd_dict[source] = xrd_vec

            self.data_dict[element.entry_id] = {'X': xrd_dict, 'Y': self.targets_saver(element.entry_id)}
