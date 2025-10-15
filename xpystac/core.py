import functools
from typing import List, Literal, Mapping, Union

import pystac
import xarray
import os
from tqdm import tqdm

from xpystac.utils import _import_optional_dependency, _is_item_search


@functools.singledispatch
def to_xarray(
    obj,
    stacking_library: Union[Literal["odc.stac", "stackstac"], None] = None,
    **kwargs,
) -> xarray.Dataset:
    """Given a PySTAC object return an xarray dataset.

    The behavior of this method depends on the type of PySTAC object:

    * Asset: if the asset points to a kerchunk file or a zarr file,
      reads the metadata in that file to construct the coordinates of the
      dataset. If the asset points to a COG, read that.
    * Item: stacks all the assets into a dataset with 1 more dimension than
      any given asset.
    * ItemCollection (output of pystac_client.search): stacks all the
      assets in all the items into a dataset with 2 more dimensions than
      any given asset.

    Parameters
    ----------
    obj : PySTAC object (Item, ItemCollection, Asset)
        The object from which to read data.
    stacking_library : "odc.stac", "stackstac", optional
        When stacking multiple items, this argument determines which library
        to use. Defaults to ``odc.stac`` if available and otherwise ``stackstac``.
    """
    if _is_item_search(obj):
        item_collection = obj.item_collection()
        return to_xarray(item_collection, stacking_library=stacking_library, **kwargs)
    raise TypeError


@to_xarray.register(pystac.Item)
@to_xarray.register(pystac.ItemCollection)
def _(
    obj: Union[pystac.Item, pystac.ItemCollection],
    drop_variables: Union[str, List[str], None] = None,
    stacking_library: Union[Literal["odc.stac", "stackstac"], None] = None,
    **kwargs,
) -> xarray.Dataset:
    if drop_variables is not None:
        raise KeyError("``drop_variables`` not implemented for pystac items")

    if stacking_library is None:
        try:
            _import_optional_dependency("odc.stac")
            stacking_library = "odc.stac"
        except ImportError:
            _import_optional_dependency("stackstac")
            stacking_library = "stackstac"
    elif stacking_library not in ["odc.stac", "stackstac"]:
        raise ValueError(f"{stacking_library=} is not a valid option")

    if stacking_library == "odc.stac":
        odc_stac = _import_optional_dependency("odc.stac")
        if isinstance(obj, pystac.Item):
            items = [obj]
        else:
            items = [i for i in obj]
        return odc_stac.load(items, **{"chunks": {"x": 1024, "y": 1024}, **kwargs})
    elif stacking_library == "stackstac":
        stackstac = _import_optional_dependency("stackstac")
        da = stackstac.stack(obj, **kwargs)
        bands = {}
        for band in da.band.values:
            b = da.sel(band=band)
            scalar_coords = {k: v.item() for k, v in b.coords.items() if v.shape == ()}
            b = b.assign_attrs(
                **{k: v for k, v in scalar_coords.items() if v is not None}
            )
            b = b.drop_vars(scalar_coords)
            bands[band] = b
        return xarray.Dataset(bands, attrs=da.attrs)


def _extract_tar_file(obj):

    tarfile=_import_optional_dependency("tarfile")
    tempfile=_import_optional_dependency("tempfile")
    pathlib=_import_optional_dependency("pathlib")
    # Extract the archive:href from obj->href ( *.tar ) to a temp dir and pass
    # this  temp dir as the obj.href to xarray
    #print(f"Encountered tar file:{obj.href}")
    tmppath=os.getenv("STAC_TMP_DIR")
    with tempfile.TemporaryDirectory(dir=tmppath,delete=False) as tmpdirname:
        print(f"tar arch ext file:{obj.extra_fields}")
        mytar = tarfile.open(obj.href)
        myvar = obj.extra_fields["archive:href"]
        myvarFiles = [ tarinfo for tarinfo in mytar.getmembers() if tarinfo.name.startswith(myvar)]
        #Check if the 'archive:href' string ends with '.zarr' .i.e., extract the entire zarr store
        if myvar.endswith(('.zarr','.zarr/')):
            #case -1 : tar of zarr .. Complete zarr store
            new_href = os.path.join(tmpdirname,obj.extra_fields["archive:href"])
            #    myvarFiles = [ tarinfo for tarinfo in mytar.getmembers() if tarinfo.name.startswith(myvar) ]
            #myvarFiles = [ tarinfo for tarinfo in mytar.getmembers() if tarinfo.name.startswith(myvar)]
        else:
            #case -2 : tar of var or #case -3 : tar of chunk .. Incomplete zarr store
            myvarFiles.extend( [ tarinfo for tarinfo in mytar.getmembers() if tarinfo.name.endswith('.zmetadata') ] )
            myvarFiles.extend( [ tarinfo for tarinfo in mytar.getmembers() if tarinfo.name.endswith('.zgroup') ] )
            zarr_path = obj.extra_fields["archive:href"].split('.zarr')[0] + '.zarr'
            new_href = os.path.join(tmpdirname,zarr_path)
        #print(myvarFiles)

        # TODO: Add a progress bar to help the user 
        mytar.extractall(tmpdirname,myvarFiles)

        #print("Extraction Done!!!")
        #print(f"tar file new:{new_href}")
    return new_href    

@to_xarray.register
def _(
    obj: Union[ pystac.Asset,list],
    stacking_library: Union[Literal["odc.stac", "stackstac"], None] = None,
    **kwargs,
) -> xarray.Dataset:

    # MKM 18 Oct 2024
    #TODO : Check if the obj is list instance or pystac.Asset instance and
    # accordingly if just one asset pass it through xarray, 
    # else, collect the assets and pass to xarray as open_mfdataset.
    # In case of tar balls, extract each of them and the store these paths
    # send the list of these zarr stores to xarray.open_mfdataset()
    # It should work.

    default_kwargs: Mapping = {"chunks": {}}

    # Check the type of the 'obj'
    if isinstance(obj, pystac.Asset):
        print("Asset as input!",flush=True)
        open_kwargs = obj.extra_fields.get("xarray:open_kwargs", {})
    
        storage_options = obj.extra_fields.get("xarray:storage_options", None)
        if storage_options:
            open_kwargs["storage_options"] = storage_options
    
        if obj.media_type == pystac.MediaType.JSON and {"index", "references"}.intersection(
            set(obj.roles) if obj.roles else set()
        ):
            requests = _import_optional_dependency("requests")
            fsspec = _import_optional_dependency("fsspec")
            r = requests.get(obj.href)
            r.raise_for_status()
            try:
                import planetary_computer  # type: ignore
    
                refs = planetary_computer.sign(r.json())
            except ImportError:
                refs = r.json()
    
            mapper = fsspec.get_mapper("reference://", fo=refs)
            default_kwargs = {
                **default_kwargs,
                "engine": "zarr",
                "consolidated": False,
            }
            return xarray.open_dataset(
                mapper, **{**default_kwargs, **open_kwargs, **kwargs}
            )
    
        if obj.media_type == pystac.MediaType.COG:
            _import_optional_dependency("rioxarray")
            default_kwargs = {**default_kwargs, "engine": "rasterio"}
        elif obj.media_type == "application/vnd+zarr":
            _import_optional_dependency("zarr")
            default_kwargs = {**default_kwargs, "engine": "zarr"}
        # MKM added for handling the 'archive' extension, as of now only plain '*.tar'
        # not zipped tarfiles.
        elif obj.media_type == "application/x-tar":
            new_href=_extract_tar_file(obj)
            print("Extraction Done!!!")
            print(f"Extracted tar file:{new_href}")
            # Check the archive:type and set the appropriate engine (as of now 'tar' of 'zarr',
            # hence 'zarr' engine) to the xarray
            default_kwargs = {**default_kwargs, "engine": "zarr"}
            # Pass the new_href and the kwargs to xarray
            return xarray.open_dataset(new_href, **{**default_kwargs, **open_kwargs, **kwargs})
        ds = xarray.open_dataset(obj.href, **{**default_kwargs, **open_kwargs, **kwargs})
        return ds

    elif isinstance(obj, list):
        print("List of Assets as input!",flush=True)
        # Create a list of assets from the list of items.
        # Prepare a dictionary to map the item, asset (tar ball) and the path where the tar ball has been extracted to.
        # Concate all the zarr stores from each tar ball and create the xarray
        # Return the xarray created above, with engine as 'zarr' ( for this particular use case )
         
        open_kwargs = obj[0].extra_fields.get("xarray:open_kwargs", {})
    
        storage_options = obj[0].extra_fields.get("xarray:storage_options", None)
        if storage_options:
            open_kwargs["storage_options"] = storage_options

        ref_media_type = obj[0].media_type
        zarr_store_list = []
        for i in tqdm(obj):
            # Check the type of the assets -- for homogenity ( all are tar balls )
            if i.media_type != ref_media_type:
                print(f"Encountered {i.to_dict()} which differs with {ref_media_type}!")
                return xarray.Dataset(data_vars=None, coords=None, attrs=None) # Empty Dataset

            if ref_media_type == "application/x-tar":
                print(f"Extracting tar file:{i.href}")
                new_href=_extract_tar_file(i)
                zarr_store_list.append(new_href)

        default_kwargs = {**default_kwargs, "engine": "zarr"}
        # Pass the new_href and the kwargs to xarray
        return xarray.open_mfdataset(zarr_store_list, **{**default_kwargs, **open_kwargs, **kwargs})

