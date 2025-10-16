import functools
from collections.abc import Callable

import pystac
import xarray
import os
from tqdm import tqdm

from xpystac._xstac_kerchunk import _stac_to_kerchunk
from xpystac.utils import _import_optional_dependency


@functools.singledispatch
def to_xarray(
    obj,
    *,
    patch_url: None | Callable[[str], str] = None,
    allow_kerchunk: bool = True,
    **kwargs,
) -> xarray.Dataset:
    """Given a PySTAC object return an xarray dataset.

    The behavior of this method depends on the type of PySTAC object:

    * Asset: if the asset points to a kerchunk file or a zarr file,
      reads the metadata in that file to construct the coordinates of the
      dataset. If the asset points to a COG, read that.
    * Item: stacks all the assets into a dataset with 1 more dimension than
      any given asset.

    Parameters
    ----------
    obj : PySTAC object (Item, ItemCollection, Asset)
        The object from which to read data.
    patch_url : Callable, optional
        Function that takes a string or pystac object and returns an altered
        version. Normally used to sign urls before trying to read data from
        them. For instance when working with Planetary Computer this argument
        should be set to ``pc.sign``.
    allow_kerchunk : bool, (True by default)
        Control whether this reader tries to interpret kerchunk attributes
        if provided (either in the data-cube extension or as a regular asset
        with ``references`` or ``index`` as the role).
    """
    raise TypeError


@to_xarray.register(pystac.Item)
def _(
    obj: pystac.Item,
    drop_variables: str | list[str] | None = None,
    patch_url: None | Callable[[str], str] = None,
    allow_kerchunk: bool = True,
    **kwargs,
) -> xarray.Dataset:
    if drop_variables is not None:
        raise KeyError("``drop_variables`` not implemented for pystac items")

    if allow_kerchunk:
        first_obj = obj if isinstance(obj, pystac.Item) else next(i for i in obj)
        is_kerchunked = any("kerchunk:" in k for k in first_obj.properties.keys())
        if is_kerchunked:
            kerchunk_combine = _import_optional_dependency("kerchunk.combine")
            fsspec = _import_optional_dependency("fsspec")

            if isinstance(obj, (list, pystac.ItemCollection)):
                refs = kerchunk_combine.MultiZarrToZarr(
                    [_stac_to_kerchunk(item) for item in obj],
                    concat_dims=kwargs.get("concat_dims", "time"),
                ).translate()
            else:
                refs = _stac_to_kerchunk(obj)

            mapper = fsspec.filesystem("reference", fo=refs).get_mapper()
            default_kwargs = {
                "engine": "zarr",
                "consolidated": False,
            }

            return xarray.open_dataset(mapper, **{**default_kwargs, **kwargs})


def _extract_tar_file(obj):

    tarfile=_import_optional_dependency("tarfile")
    tempfile=_import_optional_dependency("tempfile")
    pathlib=_import_optional_dependency("pathlib")
    # Extract the archive:href from obj->href ( *.tar ) to a temp dir and pass
    # this  temp dir as the obj.href to xarray
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
        else:
            #case -2 : tar of var or #case -3 : tar of chunk .. Incomplete zarr store
            myvarFiles.extend( [ tarinfo for tarinfo in mytar.getmembers() if tarinfo.name.endswith('.zmetadata') ] )
            myvarFiles.extend( [ tarinfo for tarinfo in mytar.getmembers() if tarinfo.name.endswith('.zgroup') ] )
            zarr_path = obj.extra_fields["archive:href"].split('.zarr')[0] + '.zarr'
            new_href = os.path.join(tmpdirname,zarr_path)

        # TODO: Add a progress bar to help the user 
        mytar.extractall(tmpdirname,myvarFiles)

    return new_href    

@to_xarray.register
def _(
    obj: pystac.Asset,
    patch_url: None | Callable[[str], str] = None,
    allow_kerchunk: bool = True,
    **kwargs,
) -> xarray.Dataset:
    open_kwargs = obj.extra_fields.get("xarray:open_kwargs", {})

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
        #open_kwargs = obj.extra_fields.get("xarray:open_kwargs", {})
    
        storage_options = obj.extra_fields.get("xarray:storage_options", None)
        if storage_options:
            open_kwargs["storage_options"] = storage_options
    
    if (
        allow_kerchunk
        and obj.media_type == pystac.MediaType.JSON
        and {"index", "references"}.intersection(set(obj.roles) if obj.roles else set())
    ):
        requests = _import_optional_dependency("requests")
        r = requests.get(obj.href)
        r.raise_for_status()

        refs = r.json()
        if patch_url is not None:
            refs = patch_url(refs)

        default_kwargs = {
            "engine": "kerchunk",
        }
        return xarray.open_dataset(refs, **{**default_kwargs, **open_kwargs, **kwargs})

    
    if obj.media_type == pystac.MediaType.COG:
        _import_optional_dependency("rioxarray")
        default_kwargs = {**default_kwargs, "engine": "rasterio"}
    elif obj.media_type in ["application/vnd+zarr", "application/vnd.zarr"]:
        _import_optional_dependency("zarr")
        zarr_kwargs = {}
        if "zarr:consolidated" in obj.extra_fields:
            zarr_kwargs["consolidated"] = obj.extra_fields["zarr:consolidated"]
        if "zarr:zarr_format" in obj.extra_fields:
            zarr_kwargs["zarr_format"] = obj.extra_fields["zarr:zarr_format"]
        default_kwargs = {**zarr_kwargs, "engine": "zarr"}
    elif obj.media_type == "application/vnd.zarr+icechunk":
        from xpystac._icechunk import read_icechunk

        return read_icechunk(obj)        
    
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
    
    href = obj.href
    if patch_url is not None:
        href = patch_url(href)

        ds = xarray.open_dataset(href, **{**default_kwargs, **open_kwargs, **kwargs})
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

